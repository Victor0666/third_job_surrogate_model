# -*- coding: utf-8 -*-
"""
resource_opt.py
说明：单位约定
- VM.pc  ：处理能力的三角模糊数（MI/s）
- VM.bw  ：带宽的三角模糊数（Mb/s）
- TriangularFuzzyNumber.modal 是原确定值，可用于与旧调度逻辑兼容

模糊化规则参考论文式 (13) 及 Section III-B：
    ξ^m = ξ
    ξ^l ~ U[δ1 ξ, ξ]
    ξ^u ~ U[ξ, min(δ2 ξ, 2ξ - ξ^l)]
论文实验参数为 δ1=0.75、δ2=1.2
"""

from dataclasses import dataclass, field
import math
import random

@dataclass(frozen=True)
class TriangularFuzzyNumber:
    """三角模糊数 ``(lower, modal, upper)``
    为兼容项目中原有的确定性训练代码，与普通数值进行运算或比较时
    使用 ``modal`` 两个三角模糊数相加时仍执行论文式 (14) 的模糊加法
    """
    lower: float
    modal: float
    upper: float

    def __post_init__(self):
        """完成 dataclass 初始化后，检查三角模糊数是否合法。"""
        values = (float(self.lower), float(self.modal), float(self.upper))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("三角模糊数的三个参数必须是有限数")
        if not self.lower <= self.modal <= self.upper:
            raise ValueError("三角模糊数必须满足 lower <= modal <= upper")

    def as_tuple(self):
        """以普通 ``(lower, modal, upper)`` 元组返回三个有限浮点分量。"""
        return self.lower, self.modal, self.upper

    def component(self, name: str) -> float:
        """显式读取一个分量，避免旧代码的 ``float(TFN)`` 模态语义产生歧义。

        参数只能是 ``"lower"``、``"modal"`` 或 ``"upper"``。这里故意不接受
        数字下标和大小写变体，使三场景传播代码能在拼写错误时立即失败。
        """
        if not isinstance(name, str):
            raise TypeError("三角模糊数分量名称必须是字符串")
        if name not in {"lower", "modal", "upper"}:
            raise ValueError(
                "三角模糊数分量名称只能是 'lower'、'modal' 或 'upper'"
            )
        return float(getattr(self, name))

    def mean(self):
        """论文式 (24) 的比例分布均值"""
        return float((self.lower + 2.0 * self.modal + self.upper) / 4.0)

    def std(self):
        """论文式 (25) 的比例分布标准差"""
        variance = (
            2.0 * (self.lower - self.modal) ** 2
            + (self.lower - self.upper) ** 2
            + 2.0 * (self.modal - self.upper) ** 2
        ) / 80.0
        return float(math.sqrt(variance))

    def score(self, uncertainty_weight=1.0):
        """论文式 (23)：均值 + 不确定性权重 × 标准差"""
        if (
            isinstance(uncertainty_weight, bool)
            or not isinstance(uncertainty_weight, (int, float))
        ):
            raise TypeError("uncertainty_weight 必须是有限非负数")
        uncertainty_weight = float(uncertainty_weight)
        if not math.isfinite(uncertainty_weight) or uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight 必须大于或等于 0")
        return float(self.mean() + uncertainty_weight * self.std())

    def fuzzy_add(self, other):
        """返回两个三角模糊数逐分量相加的结果。

        该显式接口供新的三场景评价使用；原有 ``__add__`` 行为保持不变，以免
        影响依赖 ``modal`` 标量兼容语义的 HRL/FCFS 代码。
        """
        if not isinstance(other, TriangularFuzzyNumber):
            raise TypeError("fuzzy_add 的 other 必须是 TriangularFuzzyNumber")
        return TriangularFuzzyNumber(
            self.lower + other.lower,
            self.modal + other.modal,
            self.upper + other.upper,
        )

    def fuzzy_sub(self, other):
        """按区间减法传播规则返回 ``self - other`` 的三角模糊数。"""
        if not isinstance(other, TriangularFuzzyNumber):
            raise TypeError("fuzzy_sub 的 other 必须是 TriangularFuzzyNumber")
        return TriangularFuzzyNumber(
            self.lower - other.upper,
            self.modal - other.modal,
            self.upper - other.lower,
        )

    def fuzzy_scale(self, scalar):
        """将三角模糊数乘以有限非负标量，并保持分量次序。"""
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            raise TypeError("fuzzy_scale 的 scalar 必须是有限非负数")
        scalar = float(scalar)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError("fuzzy_scale 的 scalar 必须是有限非负数")
        return TriangularFuzzyNumber(
            self.lower * scalar,
            self.modal * scalar,
            self.upper * scalar,
        )

    @classmethod
    def fuzzy_max(cls, left, right):
        """返回两个三角模糊数逐分量取最大值的结果。"""
        if not isinstance(left, cls) or not isinstance(right, cls):
            raise TypeError(
                "fuzzy_max 的 left 和 right 必须都是 TriangularFuzzyNumber"
            )
        return cls(
            max(left.lower, right.lower),
            max(left.modal, right.modal),
            max(left.upper, right.upper),
        )

    def __add__(self, other):
        """实现 ``self + other``。

        两个三角模糊数相加时，对三个对应参数分别求和并返回新的模糊数；
        与普通标量相加时仅使用当前模糊数的最可能值 ``modal``。
        """
        if isinstance(other, TriangularFuzzyNumber):
            return TriangularFuzzyNumber(
                self.lower + other.lower,
                self.modal + other.modal,
                self.upper + other.upper,
            )
        return self.modal + other

    def __radd__(self, other):
        """实现 ``other + self``，即反向加法，计算 ``other + modal``"""
        return other + self.modal

    def __sub__(self, other):
        """实现 ``self - other``，使用双方的 ``modal`` 进行确定性减法"""
        return self.modal - self._modal_of(other)

    def __rsub__(self, other):
        """实现 ``other - self``，即反向减法，计算 ``other.modal - modal``"""
        return self._modal_of(other) - self.modal

    def __mul__(self, other):
        """实现 ``self * other``，使用双方的 ``modal`` 进行确定性乘法"""
        return self.modal * self._modal_of(other)

    def __rmul__(self, other):
        """实现 ``other * self``，即反向乘法。"""
        return self._modal_of(other) * self.modal

    def __truediv__(self, other):
        """实现 ``self / other``，使用双方的 ``modal`` 进行确定性除法"""
        return self.modal / self._modal_of(other)

    def __rtruediv__(self, other):
        """实现 ``other / self``，即反向除法"""
        return self._modal_of(other) / self.modal

    def __neg__(self):
        """实现一元负号 ``-self``，返回 ``-modal``"""
        return -self.modal

    def __abs__(self):
        """实现 ``abs(self)``，返回 ``modal`` 的绝对值"""
        return abs(self.modal)

    def __float__(self):
        """实现 ``float(self)``，将最可能值 ``modal`` 转换为浮点数"""
        return float(self.modal)

    def __lt__(self, other):
        """实现 ``self < other``，比较双方的 ``modal``"""
        return self.modal < self._modal_of(other)

    def __le__(self, other):
        """实现 ``self <= other``，比较双方的 ``modal``"""
        return self.modal <= self._modal_of(other)

    def __gt__(self, other):
        """实现 ``self > other``，比较双方的 ``modal``"""
        return self.modal > self._modal_of(other)

    def __ge__(self, other):
        """实现 ``self >= other``，比较双方的 ``modal``"""
        return self.modal >= self._modal_of(other)

    @staticmethod
    def _modal_of(value):
        """若传入模糊数则提取其 ``modal``，否则直接返回原值"""
        if isinstance(value, TriangularFuzzyNumber):
            return value.modal
        return value

    @classmethod
    def zero(cls):
        """创建加法单位元，即三个参数均为 0 的三角模糊数"""
        return cls(0.0, 0.0, 0.0)


def fuzzify_resource(value, rng, delta1=0.75, delta2=1.2):
    """按照论文 Section III-B 将正的确定资源值模糊化"""
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("处理能力和带宽必须是有限正数")
    if not (0.0 < delta1 < 1.0 < delta2):
        raise ValueError("必须满足 0 < delta1 < 1 < delta2")
    if not (delta2 - 1.0 < 1.0 - delta1):
        raise ValueError("必须满足 delta2 - 1 < 1 - delta1")

    # 先随机生成下界，再限制上界，保证上下界相对最可能值不会过度偏斜：
    # upper + lower <= 2 * modal
    lower = rng.uniform(delta1 * value, value)
    upper_limit = min(delta2 * value, 2.0 * value - lower)
    upper = rng.uniform(value, upper_limit)
    return TriangularFuzzyNumber(lower, value, upper)


@dataclass
class VM:
    vm_id: int
    host_id: int
    pc: TriangularFuzzyNumber      # 模糊处理能力（MI/s）
    bw: TriangularFuzzyNumber      # 模糊带宽（Mb/s）

    def processing_time(self, workload_mi):
        """返回模糊处理时间（秒）

        时间与处理能力成反比，因此上下界顺序需要反转
        """
        workload_mi = float(workload_mi)
        if workload_mi < 0.0:
            raise ValueError("workload_mi 不能为负数")
        # 时间与处理能力成反比，因此处理能力的上下界在换算为时间时需要反转：
        # 最大处理能力对应最短时间，最小处理能力对应最长时间
        return TriangularFuzzyNumber(
            workload_mi / self.pc.upper,
            workload_mi / self.pc.modal,
            workload_mi / self.pc.lower,
        )

    def transfer_time(self, data_bits):
        """返回模糊传输时间（秒）；VM.bw 的单位为 Mb/s"""
        data_bits = float(data_bits)
        if data_bits < 0.0:
            raise ValueError("data_bits 不能为负数")
        # 带宽单位为 Mb/s，计算传输时间前先换算为 bit/s
        scale = 1e6
        return TriangularFuzzyNumber(
            data_bits / (self.bw.upper * scale),
            data_bits / (self.bw.modal * scale),
            data_bits / (self.bw.lower * scale),
        )

    def total_duration(self, workload_mi, input_bits, output_bits):
        """返回上传、计算和下载总时长的三角模糊数。

        输入/输出数据统一按 VM 模糊带宽传输；处理时间与传输时间使用显式
        ``fuzzy_add`` 逐分量相加，不依赖旧 dunder 运算的兼容行为。
        """
        input_bits = float(input_bits)
        output_bits = float(output_bits)
        if input_bits < 0.0 or output_bits < 0.0:
            raise ValueError("input_bits 和 output_bits 不能为负数")
        processing = self.processing_time(workload_mi)
        transfer = self.transfer_time(input_bits + output_bits)
        return processing.fuzzy_add(transfer)


@dataclass
class Host:
    host_id: int
    server_type: str = "cloud"
    vm_ids: list = field(default_factory=list)
    power_model: object = None
    total_pc: TriangularFuzzyNumber = field(
        default_factory=TriangularFuzzyNumber.zero
    )
    total_bw: TriangularFuzzyNumber = field(
        default_factory=TriangularFuzzyNumber.zero
    )

    def power(self, load_ratio: float) -> float:
        # 经验功耗模型仅在负载率 [0, 1] 范围内有效，因此先进行截断
        if load_ratio < 0.0:
            load_ratio = 0.0
        if load_ratio > 1.0:
            load_ratio = 1.0
        if self.power_model is None:
            return 0.0
        return self.power_model(load_ratio)


def _power_model_factory(host_id, server_type="cloud"):
    """按服务器类别和主机编号选择 SPECpower 分段线性功耗模型

    同一类别中的服务器型号按 ``host_id`` 循环分配，使集群具有异构功耗特征
    ``server_type`` 可取 ``"cloud"`` 或 ``"edge"``
    """

    def nec_express5800_gt110f_s(load: float) -> float:
        """NEC Corporation Express5800/GT110f-S 的 SPECpower 分段线性功率模型"""
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 44 + 15.9
        if load <= 0.2: return load * 21 + 18.2
        if load <= 0.3: return load * 20 + 18.4
        if load <= 0.4: return load * 28 + 16.0
        if load <= 0.5: return load * 26 + 16.8
        if load <= 0.6: return load * 32 + 13.8
        if load <= 0.7: return load * 38 + 10.2
        if load <= 0.8: return load * 27 + 17.9
        if load <= 0.9: return load * 31 + 14.7
        return load * 25 + 20.1

    def fujitsu_rx1330_m3(load: float) -> float:
        """FUJITSU PRIMERGY RX1330 M3 的 SPECpower 分段线性功率模型"""
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 44 + 13.1
        if load <= 0.2: return load * 30 + 14.5
        if load <= 0.3: return load * 25 + 15.5
        if load <= 0.4: return load * 24 + 15.8
        if load <= 0.5: return load * 28 + 14.2
        if load <= 0.6: return load * 37 + 9.7
        if load <= 0.7: return load * 54 - 0.5
        if load <= 0.8: return load * 72 - 13.1
        if load <= 0.9: return load * 70 - 11.5
        return load * 46 + 10.1
       
    def fujitsu_tx1320_m3(load: float) -> float:
        """ FUJITSU Server PRIMERGY TX1320 M3 基于 SPECpower 数据的分段线性功率模型 """
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 46.7 + 9.33
        if load <= 0.2: return load * 32.0 + 10.8
        if load <= 0.3: return load * 27.0 + 11.8
        if load <= 0.4: return load * 26.0 + 12.1
        if load <= 0.5: return load * 30.0 + 10.5
        if load <= 0.6: return load * 40.0 + 5.5
        if load <= 0.7: return load * 51.0 - 1.1
        if load <= 0.8: return load * 63.0 - 9.5
        if load <= 0.9: return load * 70.0 - 15.1
        return load * 39.0 + 12.8
    
    def lenovo_thinksystem_sr150(load: float) -> float:
        """ Lenovo Global Technology ThinkSystem SR150 基于 SPECpower 数据的分段线性功率模型 """
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 32 + 16.8
        if load <= 0.2: return load * 28 + 17.2
        if load <= 0.3: return load * 32 + 16.4
        if load <= 0.4: return load * 39 + 14.3
        if load <= 0.5: return load * 51 + 9.5
        if load <= 0.6: return load * 54 + 8.0
        if load <= 0.7: return load * 59 + 5.0
        if load <= 0.8: return load * 93 - 18.8
        if load <= 0.9: return load * 132 - 50.0
        return load * 184 - 96.8
    
    def hp_proliant_dl385_g5(load: float) -> float:
        """  HP ProLiant DL385 G5 的 SPECpower 分段线性功率模型 """
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 190 + 178
        if load <= 0.2: return load * 100 + 187
        if load <= 0.3: return load * 120 + 183
        if load <= 0.4: return load * 130 + 180
        if load <= 0.5: return load * 130 + 180
        if load <= 0.6: return load * 130 + 180
        if load <= 0.7: return load * 130 + 180
        if load <= 0.8: return load * 110 + 194
        if load <= 0.9: return load * 90 + 210
        return load * 80 + 219
    
    def hp_proliant_ml110_g4(load: float) -> float:
        """ HP ProLiant ML110 G4 的 SPECpower 分段线性功率模型 """
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 34.0 + 86.0
        if load <= 0.2: return load * 32.0 + 86.2
        if load <= 0.3: return load * 34.0 + 85.8
        if load <= 0.4: return load * 35.0 + 85.5
        if load <= 0.5: return load * 25.0 + 89.5
        if load <= 0.6: return load * 40.0 + 82.0
        if load <= 0.7: return load * 20.0 + 94.0
        if load <= 0.8: return load * 40.0 + 80.0
        if load <= 0.9: return load * 20.0 + 96.0
        return load * 30.0 + 87.0
    
    def quanta_grid_s31a_1u(load: float) -> float:
        """ QuantaGrid S31A-1U 基于 SPECpower 数据的分段线性功率模型 """
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 36.0 + 14.4
        if load <= 0.2: return load * 24.0 + 15.6
        if load <= 0.3: return load * 23.0 + 15.8
        if load <= 0.4: return load * 34.0 + 12.5
        if load <= 0.5: return load * 28.0 + 14.9
        if load <= 0.6: return load * 32.0 + 12.9
        if load <= 0.7: return load * 39.0 + 8.7
        if load <= 0.8: return load * 44.0 + 5.2
        if load <= 0.9: return load * 42.0 + 6.8
        return load * 33.0 + 14.9
    
    def fujitsu_primergy_tx120_s3p(load: float) -> float:
        """ Fujitsu PRIMERGY TX120 S3p 基于 SPECpower 数据的分段线性功率模型 """
        load = max(0.0, min(1.0, load))
        if load <= 0.1: return load * 42.0 + 15.7
        if load <= 0.2: return load * 34.0 + 16.5
        if load <= 0.3: return load * 28.0 + 17.7
        if load <= 0.4: return load * 28.0 + 17.7
        if load <= 0.5: return load * 30.0 + 16.9
        if load <= 0.6: return load * 33.0 + 15.4
        if load <= 0.7: return load * 46.0 + 7.6
        if load <= 0.8: return load * 93.0 - 25.3
        if load <= 0.9: return load * 43.0 + 14.7
        return load * 28.0 + 28.2


    
    # 分类信息与具体功耗函数集中维护；新增型号时只需加入对应元组
    models_by_server_type = {
        "cloud": (
            fujitsu_rx1330_m3,
            lenovo_thinksystem_sr150,
            nec_express5800_gt110f_s,
            hp_proliant_dl385_g5,
            hp_proliant_ml110_g4,
        ),
        "edge": (
            fujitsu_tx1320_m3,
            quanta_grid_s31a_1u,
            fujitsu_primergy_tx120_s3p,    
        ),
    }

    try:
        models = models_by_server_type[server_type]
    except KeyError as exc:
        supported_types = ", ".join(models_by_server_type)
        raise ValueError(
            f"未知服务器类别 {server_type!r}，可选类别：{supported_types}"
        ) from exc

    return models[host_id % len(models)]


def create_cluster(
    num_cloud_hosts=2,
    num_edge_hosts=1,
    cloud_vms_per_host=(3, 3),
    edge_vms_per_host=(4,),
    cloud_pc_tiers=(2.0, 4.0, 8.0),
    edge_pc_tiers=(2.0, 4.0, 8.0),
    cloud_bw_tiers=(2000.0, 4000.0, 8000.0),
    edge_bw_tiers=(2000.0, 4000.0, 8000.0),
    fuzzy_delta1=0.75,
    fuzzy_delta2=1.2,
    fuzzy_seed=0,
):
    """创建具有模糊处理能力和模糊带宽的集群

    云服务器和边缘服务器分别使用各自的 VM 数量、处理能力和带宽配置
    ``*_pc_tiers`` 和 ``*_bw_tiers`` 表示各档资源的最可能值（modal）
    ``fuzzy_seed`` 使模糊化结果可复现；传入 ``None`` 可获得非固定结果
    """
    host_groups = (
        ("cloud", num_cloud_hosts, cloud_vms_per_host, cloud_pc_tiers, cloud_bw_tiers),
        ("edge", num_edge_hosts, edge_vms_per_host, edge_pc_tiers, edge_bw_tiers),
    )

    for host_type, host_count, vms_per_host, pc_tiers, bw_tiers in host_groups:
        if not isinstance(host_count, int) or isinstance(host_count, bool):
            raise TypeError(f"{host_type} 主机数量必须是整数")
        if host_count < 0:
            raise ValueError(f"{host_type} 主机数量不能为负数")
        if not vms_per_host:
            raise ValueError(f"{host_type}_vms_per_host 不能为空")
        if len(pc_tiers) != len(bw_tiers):
            raise ValueError(
                f"{host_type}_pc_tiers 与 {host_type}_bw_tiers 必须具有相同长度"
            )
        if not pc_tiers:
            raise ValueError(f"{host_type} 资源档位不能为空")

    rng = random.Random(fuzzy_seed)
    hosts, vms = {}, {}
    vm_counter = 0
    host_id = 0
    for host_type, host_count, vms_per_host, pc_tiers, bw_tiers in host_groups:
        for type_host_id in range(host_count):
            hosts[host_id] = Host(
                host_id=host_id,
                server_type=host_type,
                power_model=_power_model_factory(type_host_id, host_type),
            )
            # 主机包含的 VM 数量和 VM 资源档位均循环使用给定模板
            vm_count_this_host = vms_per_host[type_host_id % len(vms_per_host)]
            for i in range(vm_count_this_host):
                tier = i % len(pc_tiers)
                pc = fuzzify_resource(
                    pc_tiers[tier], rng, fuzzy_delta1, fuzzy_delta2
                )
                bw = fuzzify_resource(
                    bw_tiers[tier], rng, fuzzy_delta1, fuzzy_delta2
                )
                vm = VM(vm_counter, host_id, pc, bw)
                vms[vm_counter] = vm
                hosts[host_id].vm_ids.append(vm_counter)
                # 模糊资源汇总时，分别累加下界、最可能值和上界
                hosts[host_id].total_pc = hosts[host_id].total_pc + vm.pc
                hosts[host_id].total_bw = hosts[host_id].total_bw + vm.bw
                vm_counter += 1
            host_id += 1
    return hosts, vms
