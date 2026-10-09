import os
import logging
import concurrent.futures
import time
import re
import inspect
import ast
import textwrap
import threading
import math

_api_workers = 4
_api_attempts = 3
_api_timeout = 180.0
_api_slots = threading.BoundedSemaphore(_api_workers)


class LLMRequestError(RuntimeError):
    """A temporary request failure after bounded retries."""


def init_client(cfg):
    global client
    global url
    global data
    global _api_workers, _api_attempts, _api_timeout, _api_slots
    options = getattr(cfg, "llm_api", {})
    _api_workers = int(options.get("max_parallel_requests", 4))
    _api_attempts = int(options.get("max_attempts", 3))
    _api_timeout = float(options.get("timeout", 180))
    if _api_workers < 1 or _api_attempts < 1 or not math.isfinite(_api_timeout) or _api_timeout <= 0:
        raise ValueError("llm_api limits must be positive and finite")
    _api_slots = threading.BoundedSemaphore(_api_workers)
    client_options = dict(timeout=_api_timeout, max_retries=0)
    if cfg.model.startswith("gpt"): #判断是什么大模型
        from openai import OpenAI  # 导入 OpenAI SDK
        # 检查环境变量中有没有 API Key
        assert os.getenv('OPENAI_API_KEY') is not None, "Please set the environment variable OPENAI_API_KEY"
        # OpenAI SDK also reads OPENAI_BASE_URL for compatible gateways.
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), **client_options)
        
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
            base_url="https://api.moonshot.cn/v1", **client_options
        )

    elif cfg.model.startswith("qwen"):
        from openai import OpenAI
        assert os.getenv('QWEN_API_KEY') is not None, \
            "Please set the environment variable QWEN_API_KEY"
        client = OpenAI(
            api_key=os.getenv('QWEN_API_KEY'), 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", **client_options
        )

    else:
        from openai import OpenAI
        # Default: use local or custom OpenAI-compatible API
        # 默认：使用本地或自定义的 OpenAI 兼容 API
        base_url = os.getenv('CUSTOM_API_BASE_URL', 'http://localhost:8000/v1/')
        client = OpenAI(api_key="EMPTY", base_url=base_url, **client_options)

    if cfg.model.startswith(("gpt-6.1-sol", "qwen3.8-max")):
        if "reasoning_effort" not in inspect.signature(client.chat.completions.create).parameters:
            raise RuntimeError("OpenAI SDK lacks reasoning_effort; run python -m pip install --upgrade openai")
        

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


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature, *, allow_partial=False):
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    # 如果 messages_list 不是列表的列表（即只有一段对话），则将其转换为列表的列表
    if not isinstance(messages_list, list) or not messages_list or n < 1:
        raise ValueError("messages_list must be non-empty and n must be positive")
    if not isinstance(messages_list[0], list):
        messages_list = [messages_list]
    
    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."
    
    if not model.startswith("gpt") or model.startswith("gpt-6.1-sol"):
        # Transform messages if n > 1
        # 当 n > 1 时转换消息列表
        messages_list = messages_list * n
        n = 1

    started = time.perf_counter()
    contents = [""] * (len(messages_list) * n)
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=_api_workers) as executor:
        futures = {
            executor.submit(chat_completion, n, messages, model, temperature): index
            for index, messages in enumerate(messages_list)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                choices = future.result()
            except LLMRequestError as exc:
                if not allow_partial:
                    raise
                failed += 1
                logging.warning("LLM candidate request %d failed: %s", index, exc)
                continue
            contents[index * n:(index + 1) * n] = [c.message.content for c in choices]
    logging.info("[llm batch] requests=%d failed=%d workers=%d wall_seconds=%.3f",
                 len(messages_list), failed, _api_workers, time.perf_counter() - started)
    if failed == len(messages_list):
        raise LLMRequestError("Every LLM request in this batch failed")
    return contents

def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    使用 OpenAI Chat Completions API 生成 n 个响应
    """
    if model.startswith("gpt-6.1-sol") and n > 1:
        return [choice for _ in range(n)
                for choice in chat_completion(1, messages, model, temperature)]
    last_error = None
    for attempt in range(_api_attempts):
        try:
            kwargs = dict(model=model, messages=messages)
            if model.startswith("gpt-6.1-sol"):
                kwargs.update(reasoning_effort="medium", n=1)
            elif "gpt" in model:
                kwargs.update(temperature=min(temperature, 1.), n=n)
            else:
                assert n == 1
                kwargs["temperature"] = min(temperature, 1.)
            if model.startswith("qwen3.8-max"):
                # SeEvo retains content only, not historical reasoning_content.
                kwargs.update(reasoning_effort="medium", extra_body={
                    "enable_thinking": True, "preserve_thinking": False,
                })
            with _api_slots:
                response_cur = client.chat.completions.create(**kwargs)
            if len(response_cur.choices) != n or any(
                not isinstance(c.message.content, str) or not c.message.content.strip()
                for c in response_cur.choices
            ):
                raise LLMRequestError("API returned empty content or an unexpected choice count")
            return response_cur.choices
        except Exception as e:
            status = getattr(e, "status_code", None)
            if isinstance(e, (TypeError, ValueError, AssertionError)) or (
                isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429)
            ):
                raise RuntimeError(f"Permanent LLM request error: {e}") from e
            last_error = e
            logging.warning("LLM attempt %d/%d failed: %s", attempt + 1, _api_attempts, e)
            if attempt + 1 < _api_attempts:
                delay = min(30, 2 ** attempt)
                headers = getattr(getattr(e, "response", None), "headers", {})
                try:
                    delay = max(delay, min(120, float(headers.get("retry-after", 0))))
                except (TypeError, ValueError):
                    pass
                time.sleep(delay)
    raise LLMRequestError(f"LLM request failed after {_api_attempts} attempts: {last_error}") from last_error


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
    if not isinstance(content, str) or not content.strip():
        return None
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
