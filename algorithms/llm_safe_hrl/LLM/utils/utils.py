import os
import logging
import concurrent.futures
import time
import re
import inspect
import ast
import textwrap


def init_client(cfg):
    global client
    global url
    global data
    if cfg.model.startswith("gpt"): #判断是什么大模型
        import openai
        from openai import OpenAI  # 导入 OpenAI SDK
        # 检查环境变量中有没有 API Key
        assert os.getenv('OPENAI_API_KEY') is not None, "Please set the environment variable OPENAI_API_KEY"
        # client = OpenAI(api_key=)
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        openai.api_base = "https://api.chatanywhere.tech"
        # client = OpenAI(
        #         api_key=os.getenv('OPENAI_API_KEY'),
        #         base_url="https://api.chatanywhere.tech/v1"
        #     )
        
    elif cfg.model.startswith("GLM"):
        from zhipuai import ZhipuAI
        assert os.getenv('ZHIPU_AI_API_KEY') is not None, \
            "Please set the environment variable ZHIPU_AI_API_KEY"
        client = ZhipuAI(api_key=os.getenv('ZHIPU_AI_API_KEY'))
    
    elif cfg.model.startswith("MOONSHOT"):
        from openai import OpenAI
        assert os.getenv('MOONSHOT_API_KEY') is not None, \
            "Please set the environment variable MOONSHOT_API_KEY"
        client = OpenAI(
            api_key=os.getenv('MOONSHOT_API_KEY'),
            base_url="https://api.moonshot.cn/v1"
        )

    elif cfg.model.startswith("qwen"):
        from openai import OpenAI
        assert os.getenv('QWEN_API_KEY') is not None, \
            "Please set the environment variable QWEN_API_KEY"
        client = OpenAI(
            api_key=os.getenv('QWEN_API_KEY'), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    else:
        from openai import OpenAI
        # Default: use local or custom OpenAI-compatible API
        # 默认：使用本地或自定义的 OpenAI 兼容 API
        base_url = os.getenv('CUSTOM_API_BASE_URL', 'http://localhost:8000/v1/')
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        

def file_to_string(filename: str, errors: str = "strict") -> str:
    """Read entire file content as a string.
    将整个文件内容读取为字符串。
    
    Args:
        filename: Path to the file
        filename: 文件路径
        errors: UTF-8 解码错误处理方式；普通源码保持 strict，子进程日志可用 replace
        
    Returns:
        File content as string
        字符串形式的文件内容
    """
    with open(filename, 'r', encoding='utf-8', errors=errors) as file:
        return file.read()

def filter_traceback(s: str) -> str:
    """Extract traceback error message from output string.
    从输出字符串中提取 traceback 错误信息。
    
    Args:
        s: Output string that may contain traceback
        s: 可能包含 traceback 的输出字符串
        
    Returns:
        Traceback message if found, empty string otherwise
        如果找到则返回 traceback 信息，否则返回空字符串
    """
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''

def block_until_running(stdout_filepath: str, log_status: bool = False, 
                       iter_num: int = -1, response_id: int = -1) -> None:
    """Block execution until the evaluation process has started writing output.
    阻塞执行，直到评估进程开始写入输出。
    
    Args:
        stdout_filepath: Path to the stdout file to monitor
        stdout_filepath: 需要监控的标准输出文件路径
        log_status: Whether to log execution status
        log_status: 是否记录执行状态
        iter_num: Current iteration number for logging
        iter_num: 用于日志记录的当前迭代编号
        response_id: Response ID for logging
        response_id: 用于日志记录的响应编号
    """
    while True:
        # Windows 子进程在环境变量生效前或外部程序参与时，日志中仍可能混入
        # GBK/本地代码页字节。启动监视只需要判断“已有输出/是否含 Traceback”，
        # 因此用 replace 防止解码错误把正常运行的候选误判为启动失败。
        log = file_to_string(stdout_filepath, errors="replace")
        if len(log) > 0:
            if log_status:
                if "Traceback" in log:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
                else:
                    logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    # 用于从 GPT 响应中提取代码描述的正则模式；描述以“<start>”开始，并以“<end>”结束
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature):
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    # 如果 messages_list 不是列表的列表（即只有一段对话），则将其转换为列表的列表
    assert isinstance(messages_list, list), "messages_list should be a list."
    if not isinstance(messages_list[0], list):
        messages_list = [messages_list]
    
    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."
    
    if not model.startswith(("gpt")):
        # Transform messages if n > 1
        # 当 n > 1 时转换消息列表
        messages_list *= n
        n = 1

    with concurrent.futures.ThreadPoolExecutor() as executor:
        args = [(n, messages, model, temperature) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            contents.append(c.message.content)       
    return contents

def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    使用 OpenAI Chat Completions API 生成 n 个响应
    """
    for attempt in range(1000):
        try:
            if "gpt" in model:
                response_cur = client.chat.completions.create(model=model, messages=messages, temperature = min(temperature, 1.), n=n)
            else:
                assert n == 1
                if "GLM" in model:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
                else:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
            break
        except Exception as e:
            logging.info(f"Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
    if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()
            
    return response_cur.choices


def extract_code_from_generator(content: str) -> str:
    """Extract Python code from LLM response.
    从大语言模型响应中提取 Python 代码。
    
    Args:
        content: LLM response text
        content: 大语言模型的响应文本
        
    Returns:
        Extracted Python code string or None if no valid code found
        提取出的 Python 代码字符串；如果未找到有效代码则返回 None
    """
    # Try to extract code from markdown code block
    # 尝试从 Markdown 代码块中提取代码
    pattern_code = r'```python(.*?)```'
    code_match = re.search(pattern_code, content, re.DOTALL)
    code_string = code_match.group(1).strip() if code_match else None
    
    # Seed files are plain Python rather than Markdown responses. Keep the
    # complete parseable module so nested helper returns and multiline
    # signatures cannot be truncated by line-based extraction.
    if code_string is None:
        raw_content = content.strip()
        try:
            parsed = ast.parse(raw_content)
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in parsed.body
            ):
                code_string = raw_content
        except (SyntaxError, ValueError):
            code_string = None

    # Last-resort extraction for an unfenced response containing leading prose.
    if code_string is None:
        lines = content.split('\n')
        start = next(
            (
                i for i, line in enumerate(lines)
                if line.lstrip().startswith(('def ', 'import ', 'from '))
            ),
            None,
        )
        if start is not None:
            candidate = '\n'.join(lines[start:]).strip()
            try:
                ast.parse(candidate)
                code_string = candidate
            except (SyntaxError, ValueError):
                code_string = None
    
    # Validate extracted code
    # 校验提取出的代码
    if code_string is None:
        return None
    
    if "return" not in code_string:
        return None
    
    # Add missing import statements
    # 补充缺失的导入语句
    if "np" in code_string and "import numpy" not in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string and "import torch" not in code_string:
        code_string = "import torch\n" + code_string
    
    return code_string


def filter_code(code_string: str) -> str:
    """Remove function signature and import statements from code.
    从代码中移除函数签名和导入语句。
    
    Keeps only the function body up to and including the return statement.
    仅保留函数体，直到并包含 return 语句。
    
    Args:
        code_string: Python code string
        code_string: Python 代码字符串
        
    Returns:
        Filtered code containing only the function body
        过滤后的代码，仅包含函数体
    """
    if code_string is None:
        return ""

    try:
        tree = ast.parse(code_string)
        function_node = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        body = '\n'.join(ast.unparse(node) for node in function_node.body)
        return textwrap.indent(body, '    ')
    except (SyntaxError, ValueError, StopIteration):
        pass
    
    lines = code_string.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip function definition, imports
        # 跳过函数定义和导入语句
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        # Include return statement and stop
        # 包含 return 语句并停止继续处理
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        # Include function body
        # 保留函数体内容
        else:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def get_heuristic_name(module, possible_names: list[str]) -> str:
    """Find the first function name from possible_names that exists in module.
    从 possible_names 中查找第一个存在于模块中的函数名。
    
    Args:
        module: Python module to search
        module: 需要搜索的 Python 模块
        possible_names: List of possible function names
        possible_names: 可能的函数名列表
        
    Returns:
        Name of the first matching function found, or None
        第一个匹配到的函数名；如果没有找到则返回 None
    """
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
    return None
