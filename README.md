# 联合校准分析归档

Python 3.11 标准库项目。运行工作目录为项目根目录：

```sh
python3 -B verify.py
python3 -B demo.py
```

verify 执行全部原始回归，包括真实临时 SQLite 文件重开、事务失败回滚、70 组固定种子独立有理数对照。demo 在测试临时目录构造并关闭真实旧库，重开查询，再运行独立 SDK2 原生样例；退出清理临时目录。setup 无步骤。

## 业务与精确性

模型含 1–8 个唯一非空字符串变量名；数值只接受 int、Fraction 或 Fraction 可解析的有理数字符串，拒绝 float 和 bool。均值和协方差均精确计算；协方差必须对称半正定，允许零秩、非对角奇异矩阵。没有抖动、近似逆或数值容差。

观测是至多 8 条具名线性行，输出是 1–8 条具名线性行。系数字典允许空行，缺省变量系数为零，未知变量拒绝。观测值键集必须精确匹配观测名，噪声协方差按观测字典插入顺序排列，必须对称半正定。噪声与模型独立，噪声均值为零；噪声行间允许相关、奇异及零方差。观测行与输出行的名字各自唯一，二者可同名。没有观测时传入空字典和空噪声矩阵。

Result.status 为 ok 或 inconsistent。ok 的 mean、covariance 是 Fraction 元组，表示全部输出的联合条件分布，输出顺序由 outputs 字典插入顺序决定。inconsistent 保留输出名，但 mean 和 covariance 为空元组。不在 HΣHᵀ+R 像空间内的观测残差必须被拒绝；冗余合法证据不能误拒绝。

## 旧应用公开 API

```python
from database import connect
from archive import Archive
db = connect(":memory:")  # 调用方关闭连接；文件路径也支持
a = Archive(db)
a.save_model("joint", ["X", "Y"], [0, 0], [[1, 1], [1, 1]])
handle = a.save_analysis(
    "check", "joint",
    observations={"x": {"X": 1}, "y": {"Y": 1}},
    values={"x": 0, "y": 1},
    noise=[[0, 0], [0, 0]],
    outputs={"gap": {"X": 1, "Y": -1}})
assert handle.evaluate().status == "inconsistent"
db.close()
```

Archive.save_model(model_id, variables, mean, covariance) 新建或替换模型。
save_analysis(analysis_id, model_id, observations, values, noise, outputs) 新建或替换分析，返回 Analysis 句柄。
Archive.evaluate(analysis_id) 与 Analysis(archive, analysis_id).evaluate() 等价。
load_model 返回独立 SDK1 Model；load_analysis 返回 (model_id, Request)。
模型、分析 ID 均为非空字符串，缺失 ID 抛 KeyError。
同 ID 模型替换必须保留具名变量集合，允许变量顺序改变。
保存分析只校验声明，不要求当前证据可接受；资格在 evaluate 时判断。
同 ID 分析替换可改绑模型。已持有句柄按分析 ID 读取当前绑定和当前模型，不缓存结果。
非法写入不改变已有状态；模型不提供删除接口。

samples.build_legacy(db) 接受空 SQLite 连接，完整构造 format=1 归档：
pair 为 X=Y、方差 1；offset 为 B=2A、C=-1；
full 的均值为零、协方差 [[2,1],[1,2]]；point 是确定值 3。
五条已保存分析包含合法与矛盾无噪声证据、相关奇异噪声、非零偏移与先验输出。
手算 noisy 结果均值 (8/5,-2/5)，协方差四项均为 3/5。
build_legacy 不依赖预先生成的二进制数据库或外部文件。

旧表 models_v1(id,payload)、analyses_v1(id,model_id,payload) 保存完整规范 JSON；
archive_meta 的 format 为字符串 1。噪声顺序与观测声明顺序一起持久化。
当前应用只读写旧表，经 gauss1.Model 计算。

## SDK2 原生公开端口

gauss2.Affine(variables, offset, loadings, variances) 表示
X=offset+loadings·Z，Z 的分量独立、均值零、方差为正有理数。
loadings 列必须线性独立，允许零列数的确定点模型，不需要平方根。

gauss2.Potential(variables, precision, information, constraints, values)
在 constraints·X=values 的仿射空间上定义密度
exp(-Xᵀ precision X/2 + informationᵀ X)。
precision 须半正定，并在约束的零空间内正定；允许一致冗余约束，
拒绝矛盾约束及不正规分布。没有约束时传空列表。
精度在约束外可为零；唯一确定点允许零精度和任意 information。

两个模型均支持 evaluate(Request)、payload()；
gauss2.decode(payload) 恢复并验证原生模型，不接收 SDK1 载荷。
Request 由 specs 提供，可 payload()/from_payload()，不绑定任何模型 ID。

NativeArchive(db) 只管理目标表 g2_models(id,payload)、
g2_analyses(id,model_id,payload)，不会设置 archive_meta 或访问旧表。
put_model(id,model)、get_model(id)、put_analysis(id,model_id,request)、
get_analysis(id)、evaluate(id) 是独立 SDK2 端口。
原生分析同样动态读取绑定，原生同 ID 模型替换允许跨表示并保留变量集合。
原生写入会规范化、验证并复制载荷，调用方修改原对象不会改动已存模型。

samples.affine_equal(sign) 与 potential_equal(sign) 分别手工声明
X=sign·Y、X 方差 1 的两种原生样例；sign 使用 1 或 -1。
这些端口和旧应用是两套独立归档，没有升级入口、表示转换或兼容适配。

## 事务和范围

database.atomic(db, before_commit=None) 是现成事务设施。
顶层使用 BEGIN IMMEDIATE；嵌套使用保存点，不提交调用方事务。
before_commit 在提交或释放保存点之前调用，异常回滚本次范围。
所有应用写入均复用此设施，允许外层组合模型与分析写入。
evaluate 在一个事务内读取分析和当前模型，结果与该快照一致。
database.connect 启用外键与显式事务；db 由调用方独占。
维护窗口、单工作者场景；没有在线迁移、并发写协议、拟合、非线性推断、
观测采集、测量方案优化或模型融合功能。

exact.py 提供一般有理消元、逆、行列式、核空间和 PSD 验证；
没有协方差到 SDK2 表示的构造器。PSD 验证用全部主子式，
条件分析每次完整重算，维度上界固定为 8。
测试中的独立判定器通过最大可逆主块计算期望，不调用生产线性代数或 SDK 构造期望。
