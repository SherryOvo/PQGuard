#!/usr/bin/env python3
"""
生成啤酒智慧精酿多模态知识图谱 & 智能助手的大规模模拟训练数据。

用法示例：
    .venv/bin/python generate_beer_dataset.py \
        --output data/beer_smart_brew_multimodal_kg_train_large.jsonl \
        --num-recipes 50 \
        --qas-per-recipe 8

会在保持现有 schema/字段风格的前提下，生成成百上千条实体、关系和问答样本，
供 GNN + 多模态大模型训练或指令微调使用。
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

random.seed(42)


@dataclass
class MaltSpec:
    id_prefix: str
    name: str
    base_color: float
    protein: float
    styles: List[str]


@dataclass
class HopSpec:
    id_prefix: str
    name: str
    alpha_acid: float
    aroma_tags: List[str]
    styles: List[str]


MALTS = [
    MaltSpec("pilsner", "皮尔森麦芽", 3.0, 10.5, ["浑浊IPA", "德式拉格", "美式淡艾尔"]),
    MaltSpec("wheat", "小麦麦芽", 4.0, 12.0, ["浑浊IPA", "小麦啤酒"]),
    MaltSpec("oat", "燕麦片", 2.0, 13.5, ["浑浊IPA", "世涛"]),
    MaltSpec("vienna", "维也纳麦芽", 8.0, 11.0, ["深色拉格", "琥珀艾尔"]),
]

HOPS = [
    HopSpec("citra", "Citra 啤酒花", 12.5, ["柑橘", "热带水果"], ["浑浊IPA", "美式IPA"]),
    HopSpec("mosaic", "Mosaic 啤酒花", 11.0, ["蓝莓", "松脂"], ["浑浊IPA", "淡色艾尔"]),
    HopSpec("simcoe", "Simcoe 啤酒花", 13.0, ["松针", "西柚"], ["美式IPA"]),
    HopSpec("sabro", "Sabro 啤酒花", 13.0, ["椰子", "热带水果"], ["浑浊IPA"]),
]

STYLES = [
    "浑浊IPA",
    "新英格兰IPA",
    "美式IPA",
    "淡色艾尔",
    "小麦啤酒",
]


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_schema_row() -> Dict:
    return {
        "record_type": "schema",
        "version": "v1",
        "domain": "craft_beer",
        "description": "啤酒智慧精酿多模态知识图谱与智能操作助手大规模模拟训练数据",
        "modalities": ["text", "image", "audio", "sensor"],
        "node_types": [
            "原料",
            "工艺步骤",
            "设备",
            "传感器",
            "供应商",
            "配方",
            "风味特征",
            "故障类型",
        ],
        "edge_types": [
            "使用原料",
            "采用工艺",
            "使用设备",
            "监控指标",
            "供应",
            "影响风味",
            "导致故障",
            "相似工艺",
        ],
    }


def gen_malt_entities(n_per_type: int = 5) -> List[Dict]:
    rows: List[Dict] = []
    for malt in MALTS:
        for i in range(1, n_per_type + 1):
            eid = f"malt_{malt.id_prefix}_{i}"
            color = round(malt.base_color * random.uniform(0.8, 1.3), 1)
            prot = round(malt.protein * random.uniform(0.9, 1.1), 1)
            rows.append(
                {
                    "record_type": "kg_entity",
                    "id": eid,
                    "node_type": "原料",
                    "name": f"{malt.name} 批次 {i}",
                    "attributes": {
                        "麦芽类型": "基础麦芽" if malt.id_prefix in {"pilsner", "vienna"} else "特种麦芽",
                        "颜色_EBC": color,
                        "蛋白质含量_%": prot,
                        "适用风格": malt.styles,
                        "批次号": f"{malt.id_prefix.upper()}-{20241000 + i}",
                    },
                    "modal": {
                        "text_desc": f"{malt.name} 第 {i} 批次，适用于 {', '.join(malt.styles)} 风格。",
                        "image_ref": f"images/malt/{malt.id_prefix}_{i}.jpg",
                    },
                }
            )
    return rows


def gen_hop_entities(n_per_type: int = 4) -> List[Dict]:
    rows: List[Dict] = []
    for hop in HOPS:
        for i in range(1, n_per_type + 1):
            eid = f"hop_{hop.id_prefix}_{i}"
            aa = round(hop.alpha_acid * random.uniform(0.85, 1.15), 1)
            rows.append(
                {
                    "record_type": "kg_entity",
                    "id": eid,
                    "node_type": "原料",
                    "name": f"{hop.name} 批次 {i}",
                    "attributes": {
                        "a_酸_%": aa,
                        "精油特征": hop.aroma_tags,
                        "适用风格": hop.styles,
                        "形式": "T90颗粒",
                        "批次号": f"{hop.id_prefix.upper()}-{20241000 + i}",
                    },
                    "modal": {
                        "text_desc": f"{hop.name} 批次 {i}，以 {', '.join(hop.aroma_tags)} 香气著称。",
                        "image_ref": f"images/hop/{hop.id_prefix}_{i}.jpg",
                    },
                }
            )
    return rows


def gen_recipe_entities(num_recipes: int) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(1, num_recipes + 1):
        style = random.choice(STYLES)
        abv = round(random.uniform(4.5, 8.0), 1)
        ibu = random.randint(20, 70)
        eid = f"recipe_auto_{i}"
        rows.append(
            {
                "record_type": "kg_entity",
                "id": eid,
                "node_type": "配方",
                "name": f"自动生成精酿配方 {i}（{style}）",
                "attributes": {
                    "风格": style,
                    "目标酒精度_ABV_%": abv,
                    "IBU": ibu,
                    "SRM": round(random.uniform(3, 15), 1),
                    "风味标签": ["热带水果", "柑橘", "酒体饱满"] if "IPA" in style else ["小麦", "清爽", "顺饮"],
                },
                "modal": {
                    "text_desc": f"第 {i} 号 {style} 配方，ABV≈{abv}%，IBU≈{ibu}，适合测试酿造参数推荐能力。",
                },
            }
        )
    return rows


def gen_process_entities(num_recipes: int) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(1, num_recipes + 1):
        mash_temp = random.choice([65, 66, 67, 68])
        mash_id = f"mash_auto_{i}"
        ferm_id = f"ferm_auto_{i}"
        rows.append(
            {
                "record_type": "kg_entity",
                "id": mash_id,
                "node_type": "工艺步骤",
                "name": f"单步糖化 {mash_temp}℃ 60min（配方{i}）",
                "attributes": {
                    "糖化温度_摄氏": mash_temp,
                    "持续时间_min": 60,
                    "目标可发酵度_%": random.randint(72, 82),
                    "适用配方类型": ["浑浊IPA", "淡色艾尔"],
                },
                "modal": {
                    "text_desc": f"为第 {i} 号配方设计的单步糖化方案，主控酒体与可发酵度。",
                },
            }
        )
        rows.append(
            {
                "record_type": "kg_entity",
                "id": ferm_id,
                "node_type": "工艺步骤",
                "name": f"主发酵阶段（配方{i}）",
                "attributes": {
                    "初始比重_OG": round(random.uniform(1.050, 1.075), 3),
                    "目标终点比重_FG": round(random.uniform(1.010, 1.016), 3),
                    "发酵温度_摄氏": [18, 21],
                    "发酵时间天": [5, 8],
                },
                "modal": {
                    "text_desc": f"配方 {i} 的主发酵方案，强调水果酯香与干净收口的平衡。",
                },
            }
        )
    return rows


def gen_relations_for_recipes(
    recipe_entities: List[Dict],
    malt_entities: List[Dict],
    hop_entities: List[Dict],
) -> List[Dict]:
    rows: List[Dict] = []
    for recipe in recipe_entities:
        rid = recipe["id"]
        malts = random.sample(malt_entities, k=min(3, len(malt_entities)))
        hops = random.sample(hop_entities, k=min(3, len(hop_entities)))

        # 选一个对应的工艺步骤 id
        idx = int(rid.split("_")[-1])
        mash_id = f"mash_auto_{idx}"
        ferm_id = f"ferm_auto_{idx}"

        ratio_left = 100.0
        for j, m in enumerate(malts):
            if j == len(malts) - 1:
                ratio = round(ratio_left, 1)
            else:
                ratio = round(random.uniform(10, ratio_left - 10), 1)
                ratio_left -= ratio
            rows.append(
                {
                    "record_type": "kg_relation",
                    "id": f"rel_{rid}_malt_{j+1}",
                    "edge_type": "使用原料",
                    "source_id": rid,
                    "target_id": m["id"],
                    "attributes": {"占配方麦芽比例_%": ratio},
                }
            )

        for j, h in enumerate(hops):
            rows.append(
                {
                    "record_type": "kg_relation",
                    "id": f"rel_{rid}_hop_{j+1}",
                    "edge_type": "使用原料",
                    "source_id": rid,
                    "target_id": h["id"],
                    "attributes": {
                        "使用阶段": ["旋涡增香", "干投"],
                        "干投总量_g_per_L": round(random.uniform(5.0, 12.0), 1),
                    },
                }
            )

        rows.append(
            {
                "record_type": "kg_relation",
                "id": f"rel_{rid}_mash",
                "edge_type": "采用工艺",
                "source_id": rid,
                "target_id": mash_id,
                "attributes": {"顺序": 1},
            }
        )
        rows.append(
            {
                "record_type": "kg_relation",
                "id": f"rel_{rid}_ferm",
                "edge_type": "采用工艺",
                "source_id": rid,
                "target_id": ferm_id,
                "attributes": {"顺序": 2},
            }
        )
    return rows


def gen_qa_for_recipe(recipe: Dict, num_qas: int) -> List[Dict]:
    """围绕单个配方自动生成若干 QA 样本，答案包含精确数值和详细步骤。"""
    rows: List[Dict] = []
    rid = recipe["id"]
    style = recipe["attributes"]["风格"]
    abv = recipe["attributes"]["目标酒精度_ABV_%"]
    ibu = recipe["attributes"]["IBU"]
    srm = recipe["attributes"].get("SRM", round(random.uniform(3, 15), 1))

    for i in range(num_qas):
        qa_id = f"qa_{rid}_{i+1}"
        scene = random.choice(
            [
                "工艺决策_酵母接种量",
                "原料采购_供应商推荐",
                "流程导航_糖化调整",
                "设备管理_发酵参数",
                "异常预警_发酵停滞",
            ]
        )
        
        if "IPA" in style and "决策" in scene:
            # 精确的酵母接种量和温度参数
            base_temp = random.choice([18.5, 19.0, 19.5, 20.0])
            target_temp = base_temp + random.uniform(0.3, 0.8)
            cell_count = random.choice([0.9, 1.0, 1.1, 1.2])
            user_query = f"针对配方 {rid} 的{style}，如果想让热带水果酯香更突出，应如何调整酵母接种量与发酵温度？"
            expected = (
                f"该 {style} 配方目标 ABV {abv}%、IBU {ibu}、SRM {srm}。\n"
                f"步骤1：酵母接种量调整。将标准接种量（0.75-0.85 亿细胞/mL）提升至 {cell_count} 亿细胞/mL，"
                f"使用 Wyeast 1318 London Ale III 或 Safale US-05 酵母菌株。\n"
                f"步骤2：发酵温度控制。主发酵前 48 小时将温度设定为 {base_temp}℃，"
                f"第 3-5 天逐步升温至 {target_temp}℃，并维持 {target_temp}℃ 至发酵完成。\n"
                f"步骤3：干投时机。在发酵第 4-5 天（比重降至 1.020-1.025 时）进行第一次干投，"
                f"添加 Citra 和 Mosaic 啤酒花各 {random.randint(3, 5)}g/L，α-酸含量控制在 12-14%。\n"
                f"步骤4：监控参数。每日检测比重变化，确保 72 小时内比重下降 ≥0.010，"
                f"pH 值维持在 4.2-4.5，避免产生乙酸乙酯等不良风味。"
            )
        elif "采购" in scene:
            # 具体的供应商和产品规格
            malt_protein = random.choice([11.5, 12.0, 12.5, 13.0])
            hop_aa = random.choice([12.5, 13.0, 13.5, 14.0])
            user_query = f"为配方 {rid} 的{style} 选择麦芽和啤酒花供应商，有哪些推荐？"
            expected = (
                f"针对 {style}（ABV {abv}%、IBU {ibu}）配方，推荐以下供应商和产品：\n"
                f"麦芽供应商：\n"
                f"1. 比利时 Weyermann 公司：Weyermann 皮尔森麦芽（EBC 3.5-4.0，蛋白质 {malt_protein}%），"
                f"适用于基础麦芽，提供 60-65% 的糖化收率。\n"
                f"2. 德国 Bestmalz 公司：Bestmalz 小麦麦芽（EBC 4.0-5.0，蛋白质 {malt_protein + 0.5}%），"
                f"用于浑浊 IPA，提供雾感和酒体厚度。\n"
                f"3. 美国 Briess 公司：Briess 燕麦片（蛋白质 {malt_protein + 1.0}%），"
                f"添加比例 5-10%，提升酒体顺滑度。\n"
                f"啤酒花供应商：\n"
                f"1. 美国 Yakima Chief Hops：Citra 颗粒啤酒花（α-酸 {hop_aa}%，T90 形式），"
                f"干投量 8-12g/L，提供柑橘和热带水果香气。\n"
                f"2. 美国 Hopsteiner：Mosaic 颗粒啤酒花（α-酸 {hop_aa - 0.5}%），"
                f"干投量 6-10g/L，增强蓝莓和松脂风味。\n"
                f"3. 采购建议：选择 2024 年收获批次，α-酸含量波动 ≤0.5%，"
                f"储存温度 ≤-18℃，避免光照和氧化。"
            )
        elif "流程导航" in scene:
            # 精确的糖化温度和时间
            mash_temp = random.choice([65, 66, 67, 68])
            target_temp = mash_temp + random.choice([1, 2])
            rest_time = random.choice([55, 60, 65])
            user_query = f"如果希望配方 {rid} 的{style} 酒体更饱满，糖化温度应该如何微调？"
            expected = (
                f"针对 {style}（目标 ABV {abv}%、IBU {ibu}）配方，按以下步骤调整糖化工艺：\n"
                f"步骤1：糖化温度调整。将单步糖化温度从标准 65-66℃ 提升至 {mash_temp}℃，"
                f"维持 {rest_time} 分钟，使 β-淀粉酶活性增强，产生更多可发酵糖。\n"
                f"步骤2：如需更饱满酒体，可进行两步糖化：第一步 {mash_temp}℃ 保持 {rest_time - 10} 分钟，"
                f"第二步升温至 {target_temp}℃ 保持 20 分钟，增加不可发酵糖（糊精）比例至 15-18%。\n"
                f"步骤3：糖化收率控制。目标 OG 1.060-1.075，糖化效率 ≥75%，"
                f"最终麦汁 pH 5.2-5.4，钙离子浓度 50-100 ppm。\n"
                f"步骤4：与 IBU 平衡。由于酒体增厚，需相应提高苦度，"
                f"将 IBU 从 {ibu} 调整至 {ibu + random.randint(5, 10)}，"
                f"避免甜腻感，目标 BU:GU 比例 0.8-1.0。"
            )
        elif "设备管理" in scene:
            # 具体的设备参数
            ferm_temp = random.choice([18.5, 19.0, 19.5, 20.0])
            pressure = random.choice([1.2, 1.3, 1.4, 1.5])
            warning_threshold = pressure * 0.85
            user_query = f"在当前设备条件下执行配方 {rid} 的{style}，发酵温度和压力应该如何设定更安全？"
            expected = (
                f"针对 {style}（ABV {abv}%）配方，设备参数设定如下：\n"
                f"步骤1：发酵温度设定。主发酵温度设定为 {ferm_temp}℃，"
                f"使用 PID 控制器，温度波动范围 ±0.3℃，"
                f"冷却水温度设定为 {ferm_temp - 5}℃，确保换热效率。\n"
                f"步骤2：压力控制。发酵罐工作压力设定为 {pressure} bar，"
                f"安全阀阈值 {pressure + 0.3} bar，"
                f"压力传感器预警值设定为 {warning_threshold:.2f} bar（{pressure} bar 的 85%），"
                f"当压力达到 {warning_threshold:.2f} bar 时自动发送报警。\n"
                f"步骤3：监控参数。每 4 小时记录一次温度、压力和比重，"
                f"使用数据采集系统（DAS）记录，保存至数据库，"
                f"便于后续追溯和异常分析。\n"
                f"步骤4：异常处理。如温度超过 {ferm_temp + 1}℃ 或压力超过 {pressure + 0.2} bar，"
                f"立即启动应急冷却系统，并通知操作人员。"
            )
        else:  # 异常预警
            # 详细的排查步骤和数值
            check_temp = random.choice([16.0, 16.5, 17.0, 17.5])
            target_fg = round(random.uniform(1.010, 1.016), 3)
            current_fg = target_fg + round(random.uniform(0.005, 0.010), 3)
            user_query = f"酿造配方 {rid} 的{style} 时，如果发酵曲线在两天内几乎不变化，应该如何排查发酵停滞？"
            expected = (
                f"针对 {style} 发酵停滞问题，按以下步骤排查：\n"
                f"步骤1：检查温度记录。查看最近 48 小时温度曲线，"
                f"如温度低于 {check_temp}℃（推荐温度 {check_temp + 2}℃），"
                f"立即将温度提升至 {check_temp + 2}℃，并维持 24 小时。\n"
                f"步骤2：检测比重变化。当前比重 {current_fg}，目标终点比重 {target_fg}，"
                f"如 48 小时内比重下降 <0.002，判定为发酵停滞。"
                f"检查 OG 是否为 1.060-1.075，如 OG 异常（<1.050 或 >1.080），"
                f"可能是糖化问题导致可发酵糖不足。\n"
                f"步骤3：评估酵母活性。使用显微镜检查酵母细胞密度，"
                f"正常应为 0.8-1.2 亿细胞/mL，如 <0.5 亿细胞/mL，"
                f"需重新接种，添加 0.5 亿细胞/mL 新鲜酵母（Wyeast 1318 或 Safale US-05）。\n"
                f"步骤4：检查麦汁组成。检测 pH 值（正常 4.2-4.5），"
                f"如 pH >4.8，添加食品级磷酸调节至 4.3。"
                f"检测可发酵糖含量（正常 8-12°P），如 <6°P，"
                f"可能是糖化温度过高（>70℃）导致。\n"
                f"步骤5：曝气和营养检查。确认初始溶解氧（DO）≥8 ppm，"
                f"如 DO <6 ppm，需在下一批次增加曝气时间至 20-30 分钟。"
                f"检查酵母营养盐（FAN 值），正常 180-220 mg/L，如 <150 mg/L，"
                f"添加 0.5-1.0 g/L 酵母营养盐。"
            )
        rows.append(
            {
                "record_type": "assistant_qa",
                "scene": scene,
                "id": qa_id,
                "user_query": user_query,
                "context_kg_ids": [rid],
                "expected_answer": expected,
                "modal": {
                    "text_explanation": "基于配方属性和工艺参数生成的自动化决策与指导问答样本，包含精确数值和详细步骤。",
                },
            }
        )
    return rows


def gen_multimodal_qa_samples(num_samples: int = 500) -> List[Dict]:
    """生成多模态查询样本（图像+文本、语音+文本）。"""
    rows: List[Dict] = []
    
    image_scenarios = [
        {
            "image_type": "酵母形态",
            "image_desc": "显微镜下的酵母细胞形态照片",
            "queries": [
                "这张酵母形态照片显示什么状态？",
                "根据这张酵母照片，判断酵母活性是否正常？",
                "这张图片中的酵母细胞密度是多少？",
            ],
            "answers": [
                "根据显微镜照片分析，酵母细胞呈椭圆形，大小均匀（约5-7μm），细胞壁完整，无明显空泡或变形。细胞密度约为0.9-1.1亿细胞/mL，处于正常范围。建议继续当前发酵工艺。",
                "照片显示酵母细胞形态正常，细胞膜完整，无破裂或溶解迹象。细胞出芽率约15-20%，表明酵母处于活跃增殖期。细胞密度0.95亿细胞/mL，符合接种标准。可继续发酵，无需调整。",
                "通过图像分析，当前酵母细胞密度为1.05亿细胞/mL（标准范围0.8-1.2亿细胞/mL）。细胞形态健康，出芽率18%，建议维持当前温度19.5℃，预计72小时内完成主发酵。",
            ]
        },
        {
            "image_type": "麦芽外观",
            "image_desc": "不同批次麦芽的外观对比照片",
            "queries": [
                "对比这两张麦芽照片，哪个批次质量更好？",
                "这张麦芽照片显示的颜色EBC值大约是多少？",
                "根据麦芽外观，判断是否适合浑浊IPA？",
            ],
            "answers": [
                "左侧批次麦芽颗粒饱满，颜色均匀（EBC约3.5-4.0），表面光泽良好，无霉变或异味。右侧批次颜色略深（EBC约4.5），颗粒稍小。推荐使用左侧批次，蛋白质含量约12.0%，更适合浑浊IPA。",
                "根据麦芽颜色对比卡，该批次麦芽EBC值约为3.8，属于浅色麦芽。颗粒大小均匀，蛋白质含量12.2%，符合皮尔森麦芽标准。建议用于基础麦芽，占比60-70%。",
                "该麦芽外观呈浅黄色，EBC值3.5-4.0，颗粒饱满。蛋白质含量12.5%，高于标准值（11%），有助于浑浊IPA的雾感和酒体厚度。推荐使用比例：基础麦芽60%，小麦麦芽30%，燕麦片10%。",
            ]
        },
        {
            "image_type": "发酵曲线",
            "image_desc": "发酵过程中的温度、比重变化曲线图",
            "queries": [
                "根据这张发酵曲线图，判断发酵是否正常？",
                "曲线显示温度异常，应该如何处理？",
                "从比重曲线看，发酵进度如何？",
            ],
            "answers": [
                "分析发酵曲线：温度曲线稳定在19.0-19.5℃（目标19.2℃），波动±0.2℃，正常。比重从OG 1.065在72小时内降至1.025，下降速率0.013/天，符合预期。pH从5.2降至4.3，正常。整体发酵正常，预计5-7天完成。",
                "曲线显示第36小时温度突然升至21.5℃，超出设定值19.5℃。立即检查冷却系统，将温度降至19.0℃，并维持24小时。检查比重变化，如比重下降停滞，需评估酵母活性。建议添加0.3亿细胞/mL补充酵母。",
                "比重曲线显示：0-48小时从1.065降至1.035（正常），48-72小时仅降至1.033（异常缓慢）。判定为发酵停滞。检查温度是否低于18℃（当前19.2℃正常），检查pH（当前4.4正常），建议检查酵母密度和DO值。",
            ]
        },
        {
            "image_type": "啤酒花颗粒",
            "image_desc": "T90颗粒啤酒花的外观照片",
            "queries": [
                "这张啤酒花照片显示的α-酸含量大约是多少？",
                "根据外观判断，这批啤酒花是否新鲜？",
                "这批啤酒花的储存条件是否合适？",
            ],
            "answers": [
                "根据啤酒花品种和外观特征（Citra品种，颗粒饱满，颜色鲜绿），结合包装标签信息，α-酸含量约为13.2%（标准范围12.5-14.0%）。建议干投量8-12g/L，在发酵第4-5天添加。",
                "照片显示啤酒花颗粒完整，颜色鲜绿，无褐色或发黑迹象。颗粒大小均匀，无结块。包装日期2024年9月，储存温度-18℃，符合标准。新鲜度良好，α-酸损失<5%，可正常使用。",
                "检查储存条件：包装完整，无破损。储存温度显示-18℃（标准≤-18℃），避光保存，无氧化迹象。α-酸含量13.0%，与出厂值13.2%相比损失仅1.5%，储存条件良好。建议在6个月内使用完毕。",
            ]
        },
        {
            "image_type": "糖化过程",
            "image_desc": "糖化过程中的麦汁颜色和状态照片",
            "queries": [
                "根据糖化麦汁颜色，判断糖化温度是否合适？",
                "这张糖化过程照片显示什么阶段？",
                "麦汁颜色和透明度是否正常？",
            ],
            "answers": [
                "糖化麦汁颜色呈淡黄色，透明度良好，表明糖化温度控制在65-68℃范围内。麦汁表面无泡沫异常，pH值约5.3，正常。建议继续当前糖化工艺，预计60分钟后完成。",
                "照片显示糖化进行中，麦汁颜色均匀，无分层或沉淀。温度计显示66.5℃，处于β-淀粉酶最适温度范围。麦汁流动性良好，糖化效率预计≥75%。",
                "麦汁颜色正常（EBC约8-10），透明度良好，无浑浊或悬浮物。表明糖化工艺正常，蛋白质分解充分。建议检测可发酵糖含量，目标8-12°P。",
            ]
        },
        {
            "image_type": "发酵罐内部",
            "image_desc": "发酵罐内部状态和泡沫情况照片",
            "queries": [
                "根据发酵罐内部照片，判断发酵是否正常？",
                "泡沫状态显示什么信息？",
                "发酵罐内是否有异常？",
            ],
            "answers": [
                "发酵罐内部泡沫丰富，呈白色，高度约5-8cm，表明发酵活跃。泡沫细腻，无大泡或快速破裂，说明CO2产生正常。液面颜色正常，无异常沉淀。建议继续监控，预计3-5天完成主发酵。",
                "泡沫状态显示：泡沫高度6cm，细腻均匀，无异常颜色（如褐色或黑色）。泡沫持续时间正常，表明酵母活性良好。液面有轻微波动，说明发酵正在进行。当前温度19.2℃，正常。",
                "检查发酵罐内部：无异常沉淀或悬浮物，液面颜色正常。泡沫状态健康，高度适中。无异常气味（如酸败或异味）。建议检查比重变化，确认发酵进度。",
            ]
        },
        {
            "image_type": "设备仪表盘",
            "image_desc": "发酵设备控制面板和仪表读数照片",
            "queries": [
                "根据仪表盘读数，设备参数是否正常？",
                "温度和压力读数显示什么？",
                "设备是否有报警？",
            ],
            "answers": [
                "仪表盘显示：温度19.3℃（设定19.5℃，偏差±0.2℃，正常）。压力1.35bar（设定1.4bar，正常）。冷却水流量15L/min（正常范围10-20L/min）。所有参数正常，设备运行良好。",
                "温度读数19.2℃，压力1.38bar，均在正常范围。pH显示4.3，正常。无报警指示灯，设备运行正常。建议每4小时记录一次参数，便于追溯。",
                "检查仪表盘：温度19.4℃（正常），压力1.32bar（正常），无报警。但发现冷却水流量偏低（8L/min，正常应≥10L/min），建议检查冷却系统，确保换热效率。",
            ]
        },
        {
            "image_type": "麦汁澄清度",
            "image_desc": "煮沸后麦汁的澄清度和颜色照片",
            "queries": [
                "麦汁澄清度是否达到标准？",
                "根据麦汁颜色，判断煮沸是否充分？",
                "麦汁中是否有异常沉淀？",
            ],
            "answers": [
                "麦汁澄清度良好，透明度≥90%，符合标准。颜色呈金黄色（EBC约12-14），表明煮沸充分，蛋白质凝固良好。无异常沉淀或悬浮物。建议冷却至18-20℃后转入发酵罐。",
                "麦汁颜色均匀，呈深金黄色，EBC值约13.5，表明煮沸时间充足（60-90分钟）。澄清度良好，蛋白质热凝固充分。建议检测pH（目标5.0-5.2）和可发酵糖含量。",
                "检查麦汁：澄清度正常，无异常沉淀。但发现少量蛋白质絮状物（正常现象，可通过过滤去除）。颜色正常，煮沸充分。建议使用板式换热器冷却，避免氧化。",
            ]
        },
        {
            "image_type": "原料检测报告",
            "image_desc": "麦芽和啤酒花的实验室检测报告照片",
            "queries": [
                "这份检测报告显示麦芽的蛋白质含量是多少？",
                "根据检测报告，这批原料是否符合标准？",
                "检测报告中的EBC值和α-酸含量是多少？",
            ],
            "answers": [
                "检测报告显示：麦芽蛋白质含量12.3%（标准范围11-13%），EBC值3.8，水分含量4.2%（标准≤5%），所有指标符合要求。建议用于基础麦芽，占比60-70%。",
                "报告显示：麦芽各项指标正常，蛋白质12.1%，EBC 3.6，符合皮尔森麦芽标准。啤酒花α-酸13.5%，精油含量2.1%，新鲜度良好。原料质量合格，可正常使用。",
                "检测数据：麦芽EBC 4.2，蛋白质12.8%，糖化力280WK（标准≥250WK）。啤酒花α-酸13.2%，β-酸4.8%，符合Citra品种标准。所有参数在正常范围，原料质量优秀。",
            ]
        },
        {
            "image_type": "温度分布图",
            "image_desc": "发酵罐内温度分布热力图",
            "queries": [
                "温度分布图显示是否有温度梯度？",
                "根据热力图，判断温度控制是否均匀？",
                "温度分布异常区域在哪里？",
            ],
            "answers": [
                "温度分布图显示：罐内温度均匀，中心区域19.2℃，边缘区域19.0-19.4℃，温差±0.2℃，正常。无明显的温度梯度或热点，表明搅拌和冷却系统工作正常。",
                "热力图分析：整体温度分布均匀，大部分区域在19.0-19.5℃范围内。仅在罐顶有轻微温度偏高（19.6℃），属正常现象。建议继续监控，确保温度稳定。",
                "发现温度分布异常：罐底区域温度偏低（18.5℃），可能与冷却系统有关。建议检查冷却水流量和分布，调整搅拌速度，确保温度均匀。目标：全罐温差≤0.5℃。",
            ]
        },
        {
            "image_type": "pH变化曲线",
            "image_desc": "发酵过程中pH值变化曲线图",
            "queries": [
                "pH曲线显示什么趋势？",
                "根据pH变化，判断发酵是否正常？",
                "pH值异常下降应该如何处理？",
            ],
            "answers": [
                "pH曲线显示：初始pH 5.2，48小时内降至4.4，72小时稳定在4.3，变化趋势正常。pH下降速率0.013/小时，符合预期。当前pH 4.3，处于正常范围（4.2-4.5）。",
                "分析pH曲线：0-24小时从5.2快速降至4.6（正常），24-48小时降至4.4（正常），之后稳定在4.3。整体趋势正常，无异常波动，表明发酵正常进行。",
                "pH曲线异常：48小时后pH突然降至4.0（低于正常范围4.2-4.5），可能表明酸度过高。建议检查是否有污染，检测乙酸和乳酸含量，必要时添加碳酸钙调节pH至4.3。",
            ]
        },
        {
            "image_type": "CO2产生速率",
            "image_desc": "发酵过程中CO2产生速率图表",
            "queries": [
                "CO2产生速率是否正常？",
                "根据CO2曲线，判断发酵活跃度？",
                "CO2产生突然下降说明什么？",
            ],
            "answers": [
                "CO2产生速率曲线显示：0-48小时速率持续上升至峰值（0.8L/min），48-72小时保持高位（0.7-0.8L/min），之后逐渐下降。当前速率0.6L/min，正常，表明发酵活跃。",
                "分析CO2曲线：产生速率在48小时达到峰值0.85L/min，之后稳定在0.7-0.8L/min，符合正常发酵模式。速率曲线平滑，无异常波动，表明发酵正常进行。",
                "CO2产生速率异常：72小时后突然下降至0.2L/min（正常应≥0.5L/min），可能表明发酵停滞。建议检查温度、pH和酵母活性，评估是否需要重新接种或调整工艺参数。",
            ]
        },
        {
            "image_type": "酒液颜色",
            "image_desc": "发酵完成后酒液的颜色和透明度照片",
            "queries": [
                "酒液颜色是否符合预期？",
                "根据颜色判断，这是什么风格的啤酒？",
                "酒液透明度是否正常？",
            ],
            "answers": [
                "酒液颜色呈金黄色，SRM值约6-8，符合淡色艾尔标准。透明度良好，无异常浑浊。颜色均匀，无分层，表明发酵和澄清工艺正常。建议检测最终ABV和IBU，确认是否符合配方要求。",
                "颜色分析：酒液呈深琥珀色，SRM值约12-14，符合琥珀艾尔或深色拉格特征。颜色均匀，透明度良好。结合配方信息，该颜色符合预期，表明麦芽配比和糖化工艺正确。",
                "检查酒液：颜色正常（SRM 7），但发现轻微浑浊（正常范围）。可能是酵母悬浮或蛋白质未完全沉淀。建议延长后熟时间，或使用澄清剂（如明胶）改善透明度。",
            ]
        },
        {
            "image_type": "泡沫稳定性",
            "image_desc": "倒酒后的泡沫高度和持久性照片",
            "queries": [
                "泡沫高度和持久性是否正常？",
                "根据泡沫状态，判断酒体质量？",
                "泡沫快速消失是什么原因？",
            ],
            "answers": [
                "泡沫测试：倒酒后泡沫高度3-4cm，细腻均匀，持久性良好（≥5分钟不消失）。泡沫呈白色，无异常颜色。表明蛋白质含量和CO2含量正常，酒体质量良好。",
                "泡沫分析：高度4cm，细腻，持久性6分钟，符合优质啤酒标准。泡沫结构良好，无大泡或快速破裂。结合配方，该泡沫表现符合浑浊IPA的特征，酒体质量优秀。",
                "泡沫异常：高度仅1-2cm，且快速消失（<2分钟）。可能原因：1）蛋白质含量不足；2）CO2含量偏低；3）清洁剂残留。建议检测蛋白质和CO2含量，检查清洗流程。",
            ]
        },
        {
            "image_type": "设备维护记录",
            "image_desc": "设备维护和清洁记录表照片",
            "queries": [
                "维护记录显示设备状态如何？",
                "根据记录，上次清洁是什么时候？",
                "设备是否需要维护？",
            ],
            "answers": [
                "维护记录显示：上次CIP清洗时间3天前，清洗程序完整（碱洗+酸洗+消毒），清洗效果良好。设备运行正常，无故障记录。下次维护计划7天后，当前状态良好，可正常使用。",
                "检查维护记录：上次深度清洁5天前，日常清洁每天进行。设备运行时间累计1200小时，无异常。建议按照维护计划，7天后进行深度清洁和检查，确保设备正常运行。",
                "维护记录异常：发现上次CIP清洗已超过7天（标准应≤7天），且清洗效果记录为'一般'。建议立即进行深度清洁，检查清洗系统，确保设备卫生符合标准。",
            ]
        },
    ]
    
    voice_scenarios = [
        {
            "voice_query": "语音查询：如何调整酵母接种量以提升酯香风味？",
            "text_query": "如何调整酵母接种量以提升酯香风味？",
            "answer": "针对提升酯香风味，按以下步骤调整：步骤1：将标准接种量0.75-0.85亿细胞/mL提升至1.0-1.2亿细胞/mL。步骤2：主发酵前48小时温度设定18.5℃，第3-5天逐步升温至19.5-20.0℃。步骤3：在发酵第4-5天（比重降至1.020-1.025）进行干投，添加Citra和Mosaic各4-5g/L。步骤4：每日检测比重，确保72小时内下降≥0.010，pH维持在4.2-4.5。"
        },
        {
            "voice_query": "语音查询：检测到发酵停滞，应该如何排查？",
            "text_query": "检测到发酵停滞，应该如何排查？",
            "answer": "发酵停滞排查步骤：步骤1：检查48小时温度曲线，如低于17.5℃，立即提升至19.5℃并维持24小时。步骤2：检测比重变化，如48小时内下降<0.002，判定停滞。检查OG是否为1.060-1.075。步骤3：显微镜检查酵母密度，如<0.5亿细胞/mL，重新接种0.5亿细胞/mL。步骤4：检测pH（正常4.2-4.5），如>4.8，添加磷酸调节至4.3。步骤5：检查DO≥8ppm，FAN值180-220mg/L，如不足需补充。"
        },
        {
            "voice_query": "语音查询：推荐适合浑浊IPA的麦芽供应商",
            "text_query": "推荐适合浑浊IPA的麦芽供应商",
            "answer": "浑浊IPA麦芽供应商推荐：1.比利时Weyermann公司：Weyermann皮尔森麦芽（EBC3.5-4.0，蛋白质12.0%），糖化收率60-65%。2.德国Bestmalz公司：Bestmalz小麦麦芽（EBC4.0-5.0，蛋白质12.5%），提供雾感和酒体厚度。3.美国Briess公司：Briess燕麦片（蛋白质13.0%），添加比例5-10%，提升顺滑度。采购建议：选择2024年收获批次，α-酸波动≤0.5%，储存温度≤-18℃。"
        },
        {
            "voice_query": "语音查询：糖化温度应该设定多少度？",
            "text_query": "糖化温度应该设定多少度？",
            "answer": "糖化温度设定建议：步骤1：单步糖化推荐65-68℃，维持60分钟。步骤2：如需更饱满酒体，可两步糖化：第一步65℃保持50分钟，第二步升温至68℃保持20分钟。步骤3：目标糖化效率≥75%，最终麦汁pH 5.2-5.4。步骤4：根据目标ABV和IBU调整，确保BU:GU比例0.8-1.0。"
        },
        {
            "voice_query": "语音查询：发酵罐压力异常怎么办？",
            "text_query": "发酵罐压力异常怎么办？",
            "answer": "发酵罐压力异常处理：步骤1：如压力超过1.5bar，立即检查安全阀是否正常。步骤2：检查温度是否过高（>20℃），如过高需降低至19.5℃。步骤3：检查CO2产生是否异常，如比重下降正常但压力异常，可能是设备问题。步骤4：如压力持续上升，启动应急排气，并通知维修人员。"
        },
        {
            "voice_query": "语音查询：如何判断发酵是否完成？",
            "text_query": "如何判断发酵是否完成？",
            "answer": "判断发酵完成标准：步骤1：比重连续48小时无变化，且达到目标FG（1.010-1.016）。步骤2：温度稳定，无CO2产生（泡沫消失）。步骤3：pH稳定在4.2-4.5。步骤4：酵母沉降，酒液澄清。步骤5：检测可发酵糖含量<1°P。满足以上条件即可转入后熟阶段。"
        },
        {
            "voice_query": "语音查询：干投啤酒花的最佳时机是什么时候？",
            "text_query": "干投啤酒花的最佳时机是什么时候？",
            "answer": "干投啤酒花时机：步骤1：主发酵第4-5天，比重降至1.020-1.025时进行第一次干投。步骤2：添加量8-12g/L，使用Citra、Mosaic等芳香型啤酒花。步骤3：第二次干投在发酵完成前2-3天，添加量4-6g/L。步骤4：干投后保持温度19-20℃，避免过度氧化。步骤5：干投后24-48小时检测香气，确认效果。"
        },
        {
            "voice_query": "语音查询：麦汁pH值异常如何调整？",
            "text_query": "麦汁pH值异常如何调整？",
            "answer": "麦汁pH调整方法：步骤1：目标pH 5.2-5.4，如>5.5，添加食品级磷酸或乳酸调节。步骤2：如<5.0，添加碳酸钙或氢氧化钙提升。步骤3：调整后等待10-15分钟，重新检测pH。步骤4：确保调整后pH在目标范围，避免影响酶活性。步骤5：记录调整量和最终pH，便于后续批次参考。"
        },
        {
            "voice_query": "语音查询：如何控制发酵温度？",
            "text_query": "如何控制发酵温度？",
            "answer": "发酵温度控制：步骤1：根据酵母菌株设定目标温度（通常18-20℃）。步骤2：使用PID控制器，温度波动范围±0.3℃。步骤3：冷却水温度设定为目标温度-5℃，确保换热效率。步骤4：每4小时记录温度，如异常立即调整。步骤5：主发酵前48小时保持较低温度，之后可适当升温0.5-1℃。"
        },
        {
            "voice_query": "语音查询：糖化效率低怎么办？",
            "text_query": "糖化效率低怎么办？",
            "answer": "糖化效率提升：步骤1：检查糖化温度是否在65-68℃范围内。步骤2：确保糖化时间≥60分钟。步骤3：检查麦芽粉碎度，目标：皮壳完整，胚乳充分破碎。步骤4：检测pH（目标5.2-5.4），如异常需调节。步骤5：检查水质，钙离子浓度50-100ppm有助于酶活性。如效率仍<75%，考虑更换麦芽或调整配方。"
        },
        {
            "voice_query": "语音查询：如何选择酵母菌株？",
            "text_query": "如何选择酵母菌株？",
            "answer": "酵母菌株选择：步骤1：根据啤酒风格选择，IPA推荐Wyeast 1318或Safale US-05。步骤2：考虑发酵温度范围，确保设备可控制。步骤3：评估风味特征，如需要水果酯香，选择英式艾尔酵母。步骤4：检查酒精耐受度，确保≥目标ABV。步骤5：考虑沉降性，浑浊IPA需要低沉降性酵母。建议：首次使用前进行小批量测试。"
        },
        {
            "voice_query": "语音查询：煮沸时间应该多长？",
            "text_query": "煮沸时间应该多长？",
            "answer": "煮沸时间设定：步骤1：标准煮沸时间60-90分钟，确保蛋白质凝固和DMS去除。步骤2：前15分钟添加苦味型啤酒花，计算IBU。步骤3：最后15分钟添加香味型啤酒花，保留香气。步骤4：煮沸强度：每小时蒸发量8-12%。步骤5：煮沸结束后快速冷却至18-20℃，避免DMS回溶。目标：麦汁澄清度≥90%，蛋白质凝固充分。"
        },
        {
            "voice_query": "语音查询：如何检测啤酒质量？",
            "text_query": "如何检测啤酒质量？",
            "answer": "啤酒质量检测：步骤1：检测ABV（目标±0.2%），IBU（目标±2），SRM（目标±1）。步骤2：检测pH（正常4.2-4.5），如异常可能影响稳定性。步骤3：检测可发酵糖含量（应<1°P）。步骤4：感官评价：颜色、香气、口感、苦度、酒体。步骤5：检测微生物（应无污染）。所有指标符合标准即可包装。"
        },
        {
            "voice_query": "语音查询：设备清洗流程是什么？",
            "text_query": "设备清洗流程是什么？",
            "answer": "CIP清洗流程：步骤1：预冲洗（5分钟，去除大颗粒）。步骤2：碱洗（1-2%NaOH，75-80℃，20-30分钟）。步骤3：中间冲洗（5分钟）。步骤4：酸洗（1-2%硝酸，60-70℃，15-20分钟）。步骤5：最终冲洗（10分钟，pH中性）。步骤6：消毒（过氧乙酸或蒸汽，15分钟）。清洗后检测：pH中性，无残留，微生物检测合格。"
        },
        {
            "voice_query": "语音查询：如何计算啤酒花添加量？",
            "text_query": "如何计算啤酒花添加量？",
            "answer": "啤酒花添加量计算：步骤1：根据目标IBU和α-酸含量计算苦味型啤酒花（煮沸60分钟）。公式：IBU = (α-酸% × 添加量g × 利用率%) / (麦汁体积L × 1.34)。步骤2：利用率：60分钟约30%，15分钟约15%，干投约0%。步骤3：香味型啤酒花：最后15分钟添加，量4-6g/L。步骤4：干投：8-12g/L，分2-3次添加。步骤5：根据实际α-酸含量调整，确保达到目标IBU。"
        },
        {
            "voice_query": "语音查询：后熟时间应该多长？",
            "text_query": "后熟时间应该多长？",
            "answer": "后熟时间设定：步骤1：标准后熟时间7-14天，温度0-4℃。步骤2：前3-5天：酵母沉降，CO2溶解。步骤3：5-10天：酒体成熟，风味融合。步骤4：10-14天：澄清，稳定性提升。步骤5：检测标准：酒液澄清，CO2含量≥2.5vol，无异常沉淀。如未达到标准，延长后熟时间。浑浊IPA可缩短至3-5天。"
        },
        {
            "voice_query": "语音查询：如何防止啤酒氧化？",
            "text_query": "如何防止啤酒氧化？",
            "answer": "防止氧化措施：步骤1：煮沸后快速冷却，避免热氧化。步骤2：使用板式换热器，减少与空气接触。步骤3：发酵罐充CO2保护，避免空气进入。步骤4：干投时使用密闭系统，减少氧气接触。步骤5：包装时使用CO2置换，确保顶空CO2含量≥95%。步骤6：储存温度≤4℃，避光保存。检测DO值：包装后<0.1ppm。"
        },
        {
            "voice_query": "语音查询：如何调整酒体厚度？",
            "text_query": "如何调整酒体厚度？",
            "answer": "酒体厚度调整：步骤1：糖化温度：提高至67-68℃，增加不可发酵糖（糊精）比例至15-18%。步骤2：麦芽配比：增加燕麦片或小麦麦芽比例（5-10%），提升蛋白质含量。步骤3：降低发酵度：使用低发酵度酵母，保留更多残糖。步骤4：后熟时间：适当延长，增强酒体饱满感。步骤5：与IBU平衡：酒体增厚需相应提高苦度，避免甜腻。目标：口感饱满但不黏腻。"
        },
        {
            "voice_query": "语音查询：如何检测酵母活性？",
            "text_query": "如何检测酵母活性？",
            "answer": "酵母活性检测：步骤1：显微镜检查：细胞形态正常，无空泡或变形。步骤2：细胞密度：0.8-1.2亿细胞/mL（正常范围）。步骤3：出芽率：15-25%（活跃期）。步骤4：活细胞率：≥95%（使用美蓝染色）。步骤5：发酵测试：小批量测试，24小时内比重下降≥0.010。如活性不足，需重新接种或更换酵母。"
        },
        {
            "voice_query": "语音查询：如何优化浑浊IPA的雾感？",
            "text_query": "如何优化浑浊IPA的雾感？",
            "answer": "浑浊IPA雾感优化：步骤1：麦芽配比：小麦麦芽30-40%，燕麦片10-15%，提供蛋白质和β-葡聚糖。步骤2：糖化温度：65-66℃，保留更多蛋白质。步骤3：酵母选择：低沉降性酵母（如Wyeast 1318），保持悬浮。步骤4：避免过滤：直接包装，保留酵母和蛋白质。步骤5：干投时机：发酵活跃期干投，增强雾感。目标：酒液呈浑浊状，但口感顺滑。"
        },
        {
            "voice_query": "语音查询：如何计算ABV？",
            "text_query": "如何计算ABV？",
            "answer": "ABV计算方法：步骤1：检测OG（原始比重）和FG（终点比重）。步骤2：使用公式：ABV% = (OG - FG) × 131.25。步骤3：或使用：ABV% = ((76.08 × (OG - FG)) / (1.775 - OG)) × (FG / 0.794)。步骤4：标准范围：淡色艾尔4-5%，IPA 5-7%，双倍IPA 7-9%。步骤5：检测精度：±0.2%。如ABV异常，检查糖化效率和发酵完成度。"
        },
        {
            "voice_query": "语音查询：如何判断麦芽质量？",
            "text_query": "如何判断麦芽质量？",
            "answer": "麦芽质量判断：步骤1：外观：颗粒饱满，颜色均匀，无霉变或异味。步骤2：检测指标：蛋白质11-13%，糖化力≥250WK，水分≤5%。步骤3：EBC值：符合品种标准（皮尔森3.5-4.0，小麦4.0-5.0）。步骤4：糖化收率：≥75%（标准麦芽）。步骤5：批次稳定性：α-酸波动≤0.5%，颜色波动≤0.5EBC。所有指标符合标准即为优质麦芽。"
        },
    ]
    
    # 生成图像+文本查询样本
    img_samples_needed = num_samples // 2
    img_samples_generated = 0
    
    # 首先生成所有场景的所有查询-答案组合
    base_combinations = []
    for scenario in image_scenarios:
        for query in scenario["queries"]:
            for answer in scenario["answers"]:
                base_combinations.append((scenario, query, answer))
    
    # 计算每个组合需要生成多少个变体
    combinations_count = len(base_combinations)
    variants_per_combination = max(1, img_samples_needed // combinations_count)
    remaining_samples = img_samples_needed - (variants_per_combination * combinations_count)
    
    # 为每个组合生成多个变体
    for scenario, query, answer in base_combinations:
        num_variants = variants_per_combination
        if remaining_samples > 0:
            num_variants += 1
            remaining_samples -= 1
        
        for variant_idx in range(num_variants):
            if img_samples_generated >= img_samples_needed:
                break
            rows.append({
                "record_type": "assistant_multimodal_query",
                "modality": "image+text",
                "id": f"mm_img_{len(rows)+1}",
                "user_query": query,
                "user_query_text": query,
                "image_type": scenario["image_type"],
                "image_description": scenario["image_desc"],
                "image_ref": f"images/{scenario['image_type'].lower().replace(' ', '_')}/{random.randint(1, 10000)}.jpg",
                "expected_answer": answer,
                "modal": {
                    "text_explanation": f"基于{scenario['image_type']}图像的多模态查询样本（变体{variant_idx+1}）",
                },
            })
            img_samples_generated += 1
        if img_samples_generated >= img_samples_needed:
            break
    
    # 如果还需要更多样本，随机生成
    while img_samples_generated < img_samples_needed:
        scenario = random.choice(image_scenarios)
        query = random.choice(scenario["queries"])
        answer = random.choice(scenario["answers"])
        rows.append({
            "record_type": "assistant_multimodal_query",
            "modality": "image+text",
            "id": f"mm_img_{len(rows)+1}",
            "user_query": query,
            "user_query_text": query,
            "image_type": scenario["image_type"],
            "image_description": scenario["image_desc"],
            "image_ref": f"images/{scenario['image_type'].lower().replace(' ', '_')}/{random.randint(1, 10000)}.jpg",
            "expected_answer": answer,
            "modal": {
                "text_explanation": f"基于{scenario['image_type']}图像的多模态查询样本",
            },
        })
        img_samples_generated += 1
    
    # 生成语音+文本查询样本
    voice_samples_needed = num_samples - img_samples_generated
    voice_samples_generated = 0
    
    # 计算每个语音场景需要生成多少个变体
    voice_scenarios_count = len(voice_scenarios)
    variants_per_voice_scenario = max(1, voice_samples_needed // voice_scenarios_count)
    remaining_voice_samples = voice_samples_needed - (variants_per_voice_scenario * voice_scenarios_count)
    
    # 为每个语音场景生成多个变体
    for scenario in voice_scenarios:
        num_variants = variants_per_voice_scenario
        if remaining_voice_samples > 0:
            num_variants += 1
            remaining_voice_samples -= 1
        
        for variant_idx in range(num_variants):
            if voice_samples_generated >= voice_samples_needed:
                break
            rows.append({
                "record_type": "assistant_multimodal_query",
                "modality": "voice+text",
                "id": f"mm_voice_{len(rows)+1}",
                "user_query": scenario["voice_query"],
                "user_query_text": scenario["text_query"],
                "voice_ref": f"audio/queries/{random.randint(1, 10000)}.wav",
                "expected_answer": scenario["answer"],
                "modal": {
                    "text_explanation": f"基于语音输入的多模态查询样本，包含语音转文本和文本回答（变体{variant_idx+1}）",
                },
            })
            voice_samples_generated += 1
        if voice_samples_generated >= voice_samples_needed:
            break
    
    # 如果还需要更多样本，随机生成
    while voice_samples_generated < voice_samples_needed:
        scenario = random.choice(voice_scenarios)
        rows.append({
            "record_type": "assistant_multimodal_query",
            "modality": "voice+text",
            "id": f"mm_voice_{len(rows)+1}",
            "user_query": scenario["voice_query"],
            "user_query_text": scenario["text_query"],
            "voice_ref": f"audio/queries/{random.randint(1, 10000)}.wav",
            "expected_answer": scenario["answer"],
            "modal": {
                "text_explanation": "基于语音输入的多模态查询样本，包含语音转文本和文本回答",
            },
        })
        voice_samples_generated += 1
    
    return rows


def main():
    parser = argparse.ArgumentParser(description="生成啤酒智慧精酿多模态大规模模拟数据")
    parser.add_argument(
        "--output",
        type=str,
        default="data/beer_smart_brew_multimodal_kg_train_large.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--num-recipes",
        type=int,
        default=50,
        help="生成的配方数量（每个配方会带出多条实体/关系/QA）",
    )
    parser.add_argument(
        "--qas-per-recipe",
        type=int,
        default=8,
        help="每个配方生成的 QA 个数",
    )
    parser.add_argument(
        "--num-multimodal-samples",
        type=int,
        default=None,
        help="多模态样本数量（默认：文本QA样本数量的10%，最少5000条）",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    rows: List[Dict] = []

    # 1. schema
    rows.append(build_schema_row())

    # 2. 原料实体
    malt_entities = gen_malt_entities()
    hop_entities = gen_hop_entities()
    rows.extend(malt_entities)
    rows.extend(hop_entities)

    # 3. 配方 & 工艺
    recipe_entities = gen_recipe_entities(args.num_recipes)
    process_entities = gen_process_entities(args.num_recipes)
    rows.extend(recipe_entities)
    rows.extend(process_entities)

    # 4. 关系
    rows.extend(gen_relations_for_recipes(recipe_entities, malt_entities, hop_entities))

    # 5. QA
    for recipe in recipe_entities:
        rows.extend(gen_qa_for_recipe(recipe, args.qas_per_recipe))
    
    # 6. 多模态查询样本（图像+文本、语音+文本）
    # 自动计算多模态样本数量：与文本QA样本数量匹配（80-100%）
    text_qa_count = len([r for r in rows if r.get('record_type') == 'assistant_qa'])
    if args.num_multimodal_samples is None:
        # 自动计算：文本样本的90%，确保多模态数据规模与文本数据匹配
        num_multimodal = int(text_qa_count * 0.9)
    else:
        num_multimodal = args.num_multimodal_samples
    
    print(f"生成 {num_multimodal} 条多模态样本（文本QA样本: {text_qa_count} 条，比例: {num_multimodal/text_qa_count*100:.1f}%）")
    multimodal_samples = gen_multimodal_qa_samples(num_samples=num_multimodal)
    rows.extend(multimodal_samples)

    write_jsonl(out_path, rows)
    print(f"生成完成: {len(rows)} 行 -> {out_path}")
    print(f"  - 文本QA样本: {len([r for r in rows if r.get('record_type') == 'assistant_qa'])}")
    print(f"  - 多模态样本: {len([r for r in rows if r.get('record_type') == 'assistant_multimodal_query'])}")


if __name__ == "__main__":
    main()



