from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

W = NS["w"]
R = NS["r"]
PR = NS["pr"]


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def w_attr(tag: str) -> str:
    return f"{{{W}}}{tag}"


def r_attr(tag: str) -> str:
    return f"{{{R}}}{tag}"


def pr_attr(tag: str) -> str:
    return f"{{{PR}}}{tag}"


def get_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_chapter_heading(text: str) -> bool:
    normalized = collapse_spaces(text)
    return bool(re.fullmatch(r"第[一二三四五六七八九十]+章\s+[^，。；：]{1,40}", normalized))


def is_h2_heading(text: str) -> bool:
    normalized = collapse_spaces(text)
    return bool(re.match(r"^\d+\.\d{1,2}\s+(?:[A-Za-z\u4e00-\u9fff(])", normalized))


def is_h3_heading(text: str) -> bool:
    normalized = collapse_spaces(text)
    return bool(re.match(r"^\d+\.\d{1,2}\.\d{1,2}\s+(?:[A-Za-z\u4e00-\u9fff(])", normalized))


def is_figure_caption(text: str) -> bool:
    normalized = collapse_spaces(text)
    return bool(re.match(r"^图\s*\d+(\.\d+)?\s+.+", normalized))


def is_table_caption(text: str) -> bool:
    normalized = collapse_spaces(text)
    return bool(re.match(r"^表\s*\d+(\.\d+)?\s+.+", normalized))


def ensure(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag, NS)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def remove_children(parent: ET.Element | None, tags: set[str]) -> None:
    if parent is None:
        return
    for child in list(parent):
        if child.tag in tags:
            parent.remove(child)


def set_pstyle(paragraph: ET.Element, style_id: str) -> ET.Element:
    ppr = ensure(paragraph, "w:pPr")
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        pstyle = ET.Element(qn("w", "pStyle"))
        ppr.insert(0, pstyle)
    pstyle.set(w_attr("val"), style_id)
    return ppr


def set_paragraph_indent(
    paragraph: ET.Element,
    *,
    left: str | None = None,
    right: str | None = None,
    first_line: str | None = None,
    left_chars: str | None = None,
    right_chars: str | None = None,
    first_line_chars: str | None = None,
) -> None:
    ppr = ensure(paragraph, "w:pPr")
    ind = ppr.find("w:ind", NS)
    if ind is None:
        ind = ET.SubElement(ppr, qn("w", "ind"))
    for attr_name, value in {
        "left": left,
        "right": right,
        "firstLine": first_line,
        "leftChars": left_chars,
        "rightChars": right_chars,
        "firstLineChars": first_line_chars,
    }.items():
        if value is None:
            ind.attrib.pop(w_attr(attr_name), None)
        else:
            ind.set(w_attr(attr_name), value)


def set_paragraph_spacing(
    paragraph: ET.Element,
    *,
    line: str | None = None,
    before: str | None = None,
    after: str | None = None,
    line_rule: str | None = None,
) -> None:
    ppr = ensure(paragraph, "w:pPr")
    spacing = ppr.find("w:spacing", NS)
    if spacing is None:
        spacing = ET.SubElement(ppr, qn("w", "spacing"))
    for attr_name, value in {
        "line": line,
        "before": before,
        "after": after,
        "lineRule": line_rule,
    }.items():
        if value is None:
            spacing.attrib.pop(w_attr(attr_name), None)
        else:
            spacing.set(w_attr(attr_name), value)


def clear_layout(paragraph: ET.Element, preserve_sectpr: bool = True) -> ET.Element:
    ppr = ensure(paragraph, "w:pPr")
    sectpr = ppr.find("w:sectPr", NS) if preserve_sectpr else None
    remove_children(
        ppr,
        {
            qn("w", "pStyle"),
            qn("w", "keepNext"),
            qn("w", "keepLines"),
            qn("w", "pageBreakBefore"),
            qn("w", "widowControl"),
            qn("w", "spacing"),
            qn("w", "ind"),
            qn("w", "jc"),
            qn("w", "tabs"),
            qn("w", "outlineLvl"),
            qn("w", "numPr"),
            qn("w", "rPr"),
            qn("w", "sectPr"),
        },
    )
    if preserve_sectpr and sectpr is not None:
        ppr.append(sectpr)
    return ppr


def run_elements(paragraph: ET.Element) -> list[ET.Element]:
    return paragraph.findall("w:r", NS)


def set_run_font(
    run: ET.Element,
    *,
    east_asia: str | None = None,
    ascii_font: str | None = None,
    hansi_font: str | None = None,
    size: str | None = None,
    bold: bool | None = None,
) -> None:
    rpr = ensure(run, "w:rPr")
    if east_asia is not None or ascii_font is not None or hansi_font is not None:
        fonts = rpr.find("w:rFonts", NS)
        if fonts is None:
            fonts = ET.SubElement(rpr, qn("w", "rFonts"))
        if east_asia is not None:
            fonts.set(w_attr("eastAsia"), east_asia)
        if ascii_font is not None:
            fonts.set(w_attr("ascii"), ascii_font)
        if hansi_font is not None:
            fonts.set(w_attr("hAnsi"), hansi_font)
    if size is not None:
        sz = rpr.find("w:sz", NS)
        if sz is None:
            sz = ET.SubElement(rpr, qn("w", "sz"))
        sz.set(w_attr("val"), size)
        szcs = rpr.find("w:szCs", NS)
        if szcs is None:
            szcs = ET.SubElement(rpr, qn("w", "szCs"))
        szcs.set(w_attr("val"), size)
    if bold is not None:
        existing = rpr.find("w:b", NS)
        existing_cs = rpr.find("w:bCs", NS)
        if bold:
            if existing is None:
                ET.SubElement(rpr, qn("w", "b"))
            if existing_cs is None:
                ET.SubElement(rpr, qn("w", "bCs"))
        else:
            if existing is not None:
                rpr.remove(existing)
            if existing_cs is not None:
                rpr.remove(existing_cs)


def format_cover_runs(
    paragraph: ET.Element,
    *,
    east_asia: str,
    ascii_font: str,
    size: str,
    bold: bool,
) -> None:
    for run in run_elements(paragraph):
        if run.find("w:t", NS) is None:
            continue
        set_run_font(
            run,
            east_asia=east_asia,
            ascii_font=ascii_font,
            hansi_font=ascii_font,
            size=size,
            bold=bold,
        )


def replace_paragraph_text(paragraph: ET.Element, text: str) -> None:
    for child in list(paragraph):
        if child.tag == qn("w", "r"):
            paragraph.remove(child)
    run = ET.SubElement(paragraph, qn("w", "r"))
    ET.SubElement(run, qn("w", "t")).text = text


def clear_paragraph_text_keep_drawings(paragraph: ET.Element) -> None:
    for child in list(paragraph):
        if child.tag != qn("w", "r"):
            continue
        has_drawing = (
            child.find(".//w:drawing", NS) is not None
            or child.find(".//w:object", NS) is not None
            or child.find(".//w:pict", NS) is not None
        )
        if not has_drawing:
            paragraph.remove(child)
            continue
        for text_node in list(child.findall(".//w:t", NS)):
            parent = None
            for candidate in child.iter():
                if text_node in list(candidate):
                    parent = candidate
                    break
            if parent is not None:
                parent.remove(text_node)


def make_paragraph_like(template_paragraph: ET.Element | None, text: str, *, bold: bool | None = None) -> ET.Element:
    paragraph = ET.Element(qn("w", "p"))
    if template_paragraph is not None:
        ppr = template_paragraph.find("w:pPr", NS)
        if ppr is not None:
            paragraph.append(copy.deepcopy(ppr))
    run = ET.SubElement(paragraph, qn("w", "r"))
    if template_paragraph is not None:
        template_run = template_paragraph.find("w:r", NS)
        if template_run is not None:
            rpr = template_run.find("w:rPr", NS)
            if rpr is not None:
                run.append(copy.deepcopy(rpr))
    set_run_font(run, bold=bold)
    ET.SubElement(run, qn("w", "t")).text = text
    return paragraph


def make_cell_like(template_cell: ET.Element, text: str, *, bold: bool | None = None) -> ET.Element:
    cell = ET.Element(qn("w", "tc"))
    tcpr = template_cell.find("w:tcPr", NS)
    if tcpr is not None:
        cell.append(copy.deepcopy(tcpr))
    template_paragraph = template_cell.find("w:p", NS)
    cell.append(make_paragraph_like(template_paragraph, text, bold=bold))
    return cell


def replace_table_rows(table: ET.Element, rows: list[list[str]]) -> None:
    existing_rows = table.findall("w:tr", NS)
    if not existing_rows:
        return
    header_template = existing_rows[0]
    body_template = existing_rows[1] if len(existing_rows) > 1 else existing_rows[0]
    for row in list(existing_rows):
        table.remove(row)

    def build_row(template_row: ET.Element, values: list[str], *, bold: bool | None) -> ET.Element:
        row = ET.Element(qn("w", "tr"))
        trpr = template_row.find("w:trPr", NS)
        if trpr is not None:
            row.append(copy.deepcopy(trpr))
        template_cells = template_row.findall("w:tc", NS)
        if not template_cells:
            return row
        for idx, value in enumerate(values):
            template_cell = template_cells[min(idx, len(template_cells) - 1)]
            row.append(make_cell_like(template_cell, value, bold=bold))
        return row

    table.append(build_row(header_template, rows[0], bold=True))
    for values in rows[1:]:
        table.append(build_row(body_template, values, bold=False))


def is_empty_paragraph(paragraph: ET.Element) -> bool:
    if get_text(paragraph):
        return False
    if paragraph.find(".//w:drawing", NS) is not None:
        return False
    if paragraph.find(".//w:object", NS) is not None:
        return False
    if paragraph.find(".//w:pict", NS) is not None:
        return False
    if paragraph.find(".//w:fldSimple", NS) is not None:
        return False
    return True


def import_style(target_styles: ET.Element, source_style: ET.Element) -> None:
    style_id = source_style.get(w_attr("styleId"))
    if style_id is None:
        return
    existing = target_styles.find(f"w:style[@w:styleId='{style_id}']", NS)
    if existing is not None:
        target_styles.remove(existing)
    target_styles.append(copy.deepcopy(source_style))


def update_body_text_style(styles_root: ET.Element) -> None:
    style = styles_root.find("w:style[@w:styleId='BodyText']", NS)
    if style is None:
        style = ET.SubElement(
            styles_root,
            qn("w", "style"),
            {w_attr("type"): "paragraph", w_attr("styleId"): "BodyText"},
        )
        ET.SubElement(style, qn("w", "name"), {w_attr("val"): "Body Text"})
    ppr = ensure(style, "w:pPr")
    remove_children(ppr, {qn("w", "spacing"), qn("w", "ind"), qn("w", "jc")})
    ET.SubElement(ppr, qn("w", "jc"), {w_attr("val"): "both"})
    ET.SubElement(
        ppr,
        qn("w", "spacing"),
        {
            w_attr("line"): "360",
            w_attr("lineRule"): "auto",
        },
    )
    ET.SubElement(ppr, qn("w", "ind"), {w_attr("firstLine"): "480"})
    rpr = ensure(style, "w:rPr")
    remove_children(rpr, {qn("w", "rFonts"), qn("w", "sz"), qn("w", "szCs"), qn("w", "b"), qn("w", "bCs")})
    ET.SubElement(
        rpr,
        qn("w", "rFonts"),
        {
            w_attr("ascii"): "Times New Roman",
            w_attr("hAnsi"): "Times New Roman",
            w_attr("eastAsia"): "宋体",
        },
    )
    ET.SubElement(rpr, qn("w", "sz"), {w_attr("val"): "24"})
    ET.SubElement(rpr, qn("w", "szCs"), {w_attr("val"): "24"})


def create_header(title: str, page_on_left: bool) -> bytes:
    hdr = ET.Element(qn("w", "hdr"))
    p = ET.SubElement(hdr, qn("w", "p"))
    ppr = ET.SubElement(p, qn("w", "pPr"))
    tabs = ET.SubElement(ppr, qn("w", "tabs"))
    ET.SubElement(tabs, qn("w", "tab"), {w_attr("val"): "right", w_attr("pos"): "9360"})
    ET.SubElement(ppr, qn("w", "spacing"), {w_attr("after"): "0"})

    def add_text_run(text: str) -> None:
        run = ET.SubElement(p, qn("w", "r"))
        rpr = ET.SubElement(run, qn("w", "rPr"))
        ET.SubElement(
            rpr,
            qn("w", "rFonts"),
            {
                w_attr("ascii"): "Times New Roman",
                w_attr("hAnsi"): "Times New Roman",
                w_attr("eastAsia"): "宋体",
            },
        )
        ET.SubElement(rpr, qn("w", "sz"), {w_attr("val"): "21"})
        ET.SubElement(rpr, qn("w", "szCs"), {w_attr("val"): "21"})
        t = ET.SubElement(run, qn("w", "t"))
        t.text = title if text is None else text

    def add_tab_run() -> None:
        run = ET.SubElement(p, qn("w", "r"))
        ET.SubElement(run, qn("w", "tab"))

    def add_page_field() -> None:
        fld = ET.SubElement(p, qn("w", "fldSimple"), {w_attr("instr"): "PAGE \\* MERGEFORMAT"})
        run = ET.SubElement(fld, qn("w", "r"))
        rpr = ET.SubElement(run, qn("w", "rPr"))
        ET.SubElement(
            rpr,
            qn("w", "rFonts"),
            {
                w_attr("ascii"): "Times New Roman",
                w_attr("hAnsi"): "Times New Roman",
                w_attr("eastAsia"): "宋体",
            },
        )
        ET.SubElement(rpr, qn("w", "sz"), {w_attr("val"): "21"})
        ET.SubElement(rpr, qn("w", "szCs"), {w_attr("val"): "21"})
        ET.SubElement(run, qn("w", "t")).text = "1"

    if page_on_left:
        add_page_field()
        add_tab_run()
        add_text_run(title)
    else:
        add_text_run(title)
        add_tab_run()
        add_page_field()
    return ET.tostring(hdr, encoding="utf-8", xml_declaration=True)


def build_sectpr(
    *,
    default_header_rid: str | None = None,
    even_header_rid: str | None = None,
    page_num_start: int | None = None,
    page_num_fmt: str | None = None,
    final_section: bool = False,
) -> ET.Element:
    sectpr = ET.Element(qn("w", "sectPr"))
    if default_header_rid:
        ET.SubElement(
            sectpr,
            qn("w", "headerReference"),
            {
                w_attr("type"): "default",
                r_attr("id"): default_header_rid,
            },
        )
    if even_header_rid:
        ET.SubElement(
            sectpr,
            qn("w", "headerReference"),
            {
                w_attr("type"): "even",
                r_attr("id"): even_header_rid,
            },
        )
    if not final_section:
        ET.SubElement(sectpr, qn("w", "type"), {w_attr("val"): "nextPage"})
    ET.SubElement(sectpr, qn("w", "pgSz"), {w_attr("w"): "11906", w_attr("h"): "16838"})
    ET.SubElement(
        sectpr,
        qn("w", "pgMar"),
        {
            w_attr("top"): "1701",
            w_attr("bottom"): "1134",
            w_attr("left"): "1701",
            w_attr("right"): "1134",
            w_attr("header"): "1134",
            w_attr("footer"): "567",
            w_attr("gutter"): "567",
        },
    )
    if page_num_start is not None or page_num_fmt is not None:
        attrs: dict[str, str] = {}
        if page_num_start is not None:
            attrs[w_attr("start")] = str(page_num_start)
        if page_num_fmt is not None:
            attrs[w_attr("fmt")] = page_num_fmt
        ET.SubElement(sectpr, qn("w", "pgNumType"), attrs)
    ET.SubElement(sectpr, qn("w", "cols"), {w_attr("space"): "425"})
    ET.SubElement(sectpr, qn("w", "docGrid"), {w_attr("linePitch"): "312"})
    return sectpr


def add_or_replace_sectpr(paragraph: ET.Element, sectpr: ET.Element) -> None:
    ppr = ensure(paragraph, "w:pPr")
    existing = ppr.find("w:sectPr", NS)
    if existing is not None:
        ppr.remove(existing)
    ppr.append(sectpr)


def update_settings(settings_root: ET.Element) -> None:
    if settings_root.find("w:mirrorMargins", NS) is None:
        settings_root.insert(1, ET.Element(qn("w", "mirrorMargins")))
    if settings_root.find("w:evenAndOddHeaders", NS) is None:
        settings_root.insert(2, ET.Element(qn("w", "evenAndOddHeaders")))


def cleanup_figure_artifacts(body: ET.Element) -> None:
    paragraphs = [child for child in list(body) if child.tag == qn("w", "p")]
    by_text = {get_text(paragraph): paragraph for paragraph in paragraphs if get_text(paragraph)}

    remove_exact = {
        "系统总体框架图",
        "数据集构成统计图与统计表",
        "训练/验证样本与检测框规模",
        "clear     样 本",
        "大气散射模型",
        "/(x)=J(x)·t(x)+A·(1-t(x))",
        "t(x)=e-     Bd(x)",
        "在线造雾原理图与三类天气样例图",
        "清晰原图",
        "uniform 样本",
        "深度图(伪彩)",
        "patchy 样本",
        "patchy    模式通过低频噪声调制有效深度，",
        "从而模拟局地浓淡不均的团雾。",
        "统一多任务模型结构图",
        "输 出",
        "推理结果可视化图(最终演示视频帧)",
        "混合推理右下角“雾浓度分布图”示例",
        "训练与验证损失曲线图",
        "2.50",
        "2.25",
        "1.50",
        "1.25",
        "1.00",
        "5.0    5.5",
        "总损失曲线",
        "8.06.0               6.5       7.0    7.5 Epoch",
        "训练总损失 验证总损失",
        "8.5                  9.0",
        "雾分类混淆矩阵",
        "雾分类混淆矩阵(合成验证样本)",
        "1.0",
        "clear",
        "100.0%",
        "0",
        "0.0%",
        "0.80   0.0%",
        "0.6",
        "0.0",
        "0                                          0                          70",
        "0.0%                       0.0%                      100.0%",
        "0.4",
        "patchy",
        "0.2 126",
        "clear                     uniform                     patchy",
        "预测类别",
    }
    remove_contains = [
        "FogAusmenfAionXML",
        "检 测 分 支YOLO",
        "训练阶段子损失曲线检测损失雾分类损失beta",
        "不同训练轮次下 beta   分布图0.070.060.050.040.",
        "UA-DETRAC  验证集上的典型漏检与误检案例 (fogfocus",
    ]
    clear_text_exact = {
        "数量数量",
        "训练序列                测试序列                XML. 文件",
        "真 实 类 别真 实 类 别uniform",
        "cus full)典型误检案例MVI_63521/img00255.jpg典型漏检案例 MVI_63521/img00336.jpg",
        "GT 车辆框    模型保留检测框    漏检目标    误检目标",
    }

    for paragraph in list(body):
        if paragraph.tag != qn("w", "p"):
            continue
        text = get_text(paragraph)
        if not text:
            continue
        if text in remove_exact or any(token in text for token in remove_contains):
            body.remove(paragraph)
            continue
        if text in clear_text_exact:
            clear_paragraph_text_keep_drawings(paragraph)

    paragraphs = [child for child in list(body) if child.tag == qn("w", "p")]
    split_head = None
    split_tail = None
    for paragraph in paragraphs:
        text = get_text(paragraph)
        if text.startswith("为使表 7.8 的数值结果更具可读性"):
            split_head = paragraph
        elif text == "标边界不完整或相邻车辆结构相互干扰的区域。":
            split_tail = paragraph
    if split_head is not None:
        replace_paragraph_text(
            split_head,
            "为使表 7.8 的数值结果更具可读性，图 7.5进一步给出了 fogfocus_full 权重在 UA-DETRAC 验证集上的一例典型漏检和一例典型误检。左图中，高亮框对应被漏检的车辆目标；右图中，高亮框对应高置信误检区域。可以看到，这些错误往往并非发生在目标完全不可见的场景，而是更容易出现在遮挡、透视形变、目标边界不完整或相邻车辆结构相互干扰的区域。",
        )
    if split_tail is not None and split_tail in list(body):
        body.remove(split_tail)


def normalize_target_tables(body: ET.Element) -> None:
    table_specs = [
        {
            "caption_contains": "表 7.3",
            "rows": [
                ["项目", "取值"],
                ["起始 checkpoint", "outputs/Fog_Detection_Project_fogfocus/checkpoints/checkpoint_epoch_0004.pt"],
                ["输出目录", "outputs/Fog_Detection_Project_fogfocus_full"],
                ["总 epoch 数", "12"],
                ["学习率", "1×10^-5"],
                ["FREEZE_YOLO_FOR_FOG", "1"],
                ["DET_LOSS_WEIGHT", "0.0"],
                ["FOG_CLS_LOSS_WEIGHT", "1.75"],
                ["FOG_REG_LOSS_WEIGHT", "1.35"],
                ["MAX_TRAIN_BATCHES / MAX_VAL_BATCHES", "0 / 0（不做 smoke 限制）"],
                ["SKIP_QAT", "1"],
            ],
            "remove_prefixes": [
                "起始 checkpoint",
                "输出目录",
                "学习率",
                "FREEZE_YOLO_FOR_FOG",
                "DET_LOSS_WEIGHT",
                "FOG_CLS_LOSS_WEIGHT",
                "FOG_REG_LOSS_WEIGHT",
                "MAX_TRAIN_BATCHES",
                "/ MAX_VAL_BATCHES",
                "SKIP_QAT",
            ],
        },
        {
            "caption_contains": "表7.4",
            "rows": [
                ["指标", "数值"],
                ["最佳验证损失", "1.0072"],
                ["第 12 个 epoch 训练总损失", "0.9346"],
                ["第 12 个 epoch 验证总损失", "1.0072"],
                ["训练阶段 fog classification loss", "0.5340"],
                ["验证阶段 fog classification loss", "0.5754"],
                ["训练阶段 fog regression loss", "1.322×10^-4"],
                ["是否出现 non-finite gradient", "否"],
                ["是否触发 AMP recovery", "否"],
            ],
            "remove_prefixes": [],
        },
        {
            "table_header_contains": "方案",
            "rows": [
                ["路线", "fog 权重", "车辆检测器", "统一方案平均检测数/帧", "非零检测帧占比", "mean beta", "dominant fog class", "结论"],
                ["默认统一模型", "outputs/Fog_Detection_Project/unified_model_best.pt", "统一模型检测头", "0.411", "0.411", "0.00570", "CLEAR", "不推荐作为最终结果"],
                ["视频适配统一模型", "outputs/Fog_Detection_Project_videoadapt_formal/unified_model_best.pt", "统一模型检测头", "0.411", "0.411", "0.00575", "CLEAR", "未改善 clear 偏置"],
                ["偏雾天统一模型", "outputs/Fog_Detection_Project_fogfocus/unified_model_best.pt", "统一模型检测头", "0.685", "0.571", "0.06782", "PATCHY FOG", "可作为阶段性最佳统一雾权重"],
                ["Fog-Focused 正式训练后统一模型", "outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt", "统一模型检测头", "0.685", "0.571", "0.06709", "UNIFORM FOG", "正式 fog 权重，更适合论文正式结果"],
                ["最终演示混合方案", "outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt", "yolo11n.pt", "0.685（统一侧） / 1.768（混合侧）", "0.571（统一侧） / 0.488（混合侧）", "0.06709", "UNIFORM FOG", "最终采用"],
            ],
            "remove_prefixes": [],
        },
        {
            "caption_contains": "表7.8",
            "rows": [
                ["方法", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "说明"],
                ["默认 unified 模型", "0.8836", "0.8038", "0.8937", "0.6570", "outputs/Fog_Detection_Project/unified_model_best.pt"],
                ["fogfocus 权重", "0.8720", "0.7700", "0.8645", "0.6204", "outputs/Fog_Detection_Project_fogfocus/unified_model_best.pt"],
                ["fogfocus_full 权重", "0.8720", "0.7700", "0.8645", "0.6204", "outputs/Fog_Detection_Project_fogfocus_full/unified_model_best.pt"],
                ["yolo11n 车辆基线", "0.8077", "0.7344", "0.8102", "0.5714", "最终混合路线所使用的独立检测器"],
            ],
            "remove_prefixes": [],
        },
    ]

    children = list(body)
    for spec in table_specs:
        caption_idx = None
        table_idx = None
        if "caption_contains" in spec:
            for idx, child in enumerate(children):
                if child.tag == qn("w", "p") and spec["caption_contains"] in get_text(child):
                    caption_idx = idx
                    break
            if caption_idx is None:
                continue
            for idx in range(caption_idx + 1, len(children)):
                if children[idx].tag == qn("w", "tbl"):
                    table_idx = idx
                    break
            if table_idx is None:
                continue
        else:
            for idx, child in enumerate(children):
                if child.tag != qn("w", "tbl"):
                    continue
                header_row = child.find("w:tr", NS)
                if header_row is None:
                    continue
                header_text = " ".join(get_text(tc) for tc in header_row.findall("w:tc", NS))
                if spec["table_header_contains"] in header_text:
                    table_idx = idx
                    break
            if table_idx is None:
                continue

        replace_table_rows(children[table_idx], spec["rows"])

        if not spec["remove_prefixes"]:
            continue

        cursor = table_idx + 1
        while cursor < len(children):
            child = children[cursor]
            if child.tag != qn("w", "p"):
                break
            text = get_text(child)
            if not text:
                body.remove(child)
                children.pop(cursor)
                continue
            if any(text.startswith(prefix) for prefix in spec["remove_prefixes"]):
                body.remove(child)
                children.pop(cursor)
                continue
            break


def main() -> None:
    root = Path("/root/BS")
    target_path = root / "毕业设计.docx"
    template_path = root / "本科毕业设计论文范文.docx"
    backup_path = root / "毕业设计.before_format_fix_20260429.docx"

    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)

    with zipfile.ZipFile(target_path, "r") as zf:
        file_map = {info.filename: zf.read(info.filename) for info in zf.infolist()}

    with zipfile.ZipFile(template_path, "r") as zf:
        template_styles = ET.fromstring(zf.read("word/styles.xml"))

    document_root = ET.fromstring(file_map["word/document.xml"])
    styles_root = ET.fromstring(file_map["word/styles.xml"])
    settings_root = ET.fromstring(file_map["word/settings.xml"])
    rels_root = ET.fromstring(file_map["word/_rels/document.xml.rels"])

    for style_id in ["a0", "a1", "a4", "aff4", "aff7", "1", "2", "3", "a", "-1", "-", "a5"]:
        source = template_styles.find(f"w:style[@w:styleId='{style_id}']", NS)
        if source is not None:
            import_style(styles_root, source)
    update_body_text_style(styles_root)
    update_settings(settings_root)

    body = document_root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Missing document body")

    top_paragraphs = [child for child in list(body) if child.tag == qn("w", "p")]

    text_to_paragraph: list[tuple[int, ET.Element, str]] = [
        (index, paragraph, get_text(paragraph)) for index, paragraph in enumerate(top_paragraphs, start=1)
    ]

    cn_abs = next(
        p
        for _, p, text in text_to_paragraph
        if collapse_spaces(text.replace("摘   要", "摘 要")).replace(" ", "") in {"摘要", "摘要摘要摘要"}
    )
    en_abs = next(p for _, p, text in text_to_paragraph if collapse_spaces(text).upper() == "ABSTRACT")
    refs_title = next(p for _, p, text in text_to_paragraph if collapse_spaces(text) == "参考文献")
    chapter_starts = [p for _, p, text in text_to_paragraph if is_chapter_heading(text)]

    section_starts = [cn_abs, en_abs, *chapter_starts, refs_title]

    def find_holder(start_para: ET.Element) -> ET.Element:
        start_idx = top_paragraphs.index(start_para)
        for idx in range(start_idx - 1, -1, -1):
            candidate = top_paragraphs[idx]
            if is_empty_paragraph(candidate):
                return candidate
        return top_paragraphs[start_idx - 1]

    boundary_holders = [find_holder(start_para) for start_para in section_starts]
    preserve_empty = {
        paragraph
        for index, paragraph, _ in text_to_paragraph
        if index < top_paragraphs.index(cn_abs) + 1 and index <= 25 and is_empty_paragraph(paragraph)
    }
    preserve_empty.update(boundary_holders)

    for paragraph in list(body):
        if paragraph.tag != qn("w", "p"):
            continue
        if paragraph in preserve_empty:
            continue
        if is_empty_paragraph(paragraph):
            body.remove(paragraph)

    top_paragraphs = [child for child in list(body) if child.tag == qn("w", "p")]

    for paragraph in top_paragraphs:
        ppr = ensure(paragraph, "w:pPr")
        sectpr = ppr.find("w:sectPr", NS)
        if sectpr is not None:
            ppr.remove(sectpr)
        clear_layout(paragraph, preserve_sectpr=False)

    trailing_sectpr = body.find("w:sectPr", NS)
    if trailing_sectpr is not None:
        body.remove(trailing_sectpr)

    cover_end_idx = top_paragraphs.index(boundary_holders[0])
    cover_paragraphs = set(top_paragraphs[:cover_end_idx])

    references_started = False
    for paragraph in top_paragraphs:
        text = get_text(paragraph)
        normalized = collapse_spaces(text)
        if not normalized:
            set_paragraph_spacing(paragraph, before="0", after="0")
            continue

        if paragraph in cover_paragraphs:
            clear_layout(paragraph)
            if normalized.startswith("班") or normalized.startswith("学号"):
                set_paragraph_indent(
                    paragraph,
                    left="5250",
                    right="840",
                    left_chars="2500",
                    right_chars="400",
                )
                format_cover_runs(
                    paragraph,
                    east_asia="宋体",
                    ascii_font="Times New Roman",
                    size="24",
                    bold=True,
                )
            elif normalized == "本科毕业设计论文":
                set_paragraph_indent(paragraph, first_line="422", first_line_chars="44")
                format_cover_runs(
                    paragraph,
                    east_asia="黑体",
                    ascii_font="Times New Roman",
                    size="84",
                    bold=False,
                )
            elif normalized.startswith("题"):
                set_paragraph_indent(paragraph, first_line="993", first_line_chars="309")
                format_cover_runs(
                    paragraph,
                    east_asia="宋体",
                    ascii_font="Times New Roman",
                    size="32",
                    bold=True,
                )
            elif normalized.startswith(("学院", "学 院", "学")) and "院" in normalized[:6]:
                set_paragraph_indent(paragraph, first_line="993", first_line_chars="309")
                format_cover_runs(
                    paragraph,
                    east_asia="宋体",
                    ascii_font="Times New Roman",
                    size="24",
                    bold=True,
                )
            elif normalized.startswith(("专", "学生姓名", "学 生 姓 名", "导师姓名", "导 师 姓 名")):
                set_paragraph_indent(paragraph, first_line="993", first_line_chars="309")
                format_cover_runs(
                    paragraph,
                    east_asia="宋体",
                    ascii_font="Times New Roman",
                    size="24",
                    bold=True,
                )
            else:
                set_paragraph_indent(paragraph, first_line="927", first_line_chars="309")
                format_cover_runs(
                    paragraph,
                    east_asia="宋体",
                    ascii_font="Times New Roman",
                    size="32",
                    bold=True,
                )
            continue

        clear_layout(paragraph)

        if paragraph is cn_abs:
            replace_paragraph_text(paragraph, "摘  要")
            set_pstyle(paragraph, "1")
        elif paragraph is en_abs:
            replace_paragraph_text(paragraph, "Abstract")
            set_pstyle(paragraph, "1")
        elif normalized == "参考文献":
            set_pstyle(paragraph, "1")
            references_started = True
        elif is_chapter_heading(normalized):
            set_pstyle(paragraph, "1")
        elif is_h3_heading(normalized):
            set_pstyle(paragraph, "3")
        elif is_h2_heading(normalized):
            set_pstyle(paragraph, "2")
        elif is_figure_caption(normalized):
            set_pstyle(paragraph, "-1")
        elif is_table_caption(normalized):
            set_pstyle(paragraph, "-")
        elif normalized.startswith("关键词：") or normalized.startswith("Keywords:"):
            set_pstyle(paragraph, "BodyText")
            set_paragraph_indent(paragraph, first_line=None)
        elif references_started and normalized.startswith("["):
            set_pstyle(paragraph, "a")
        else:
            set_pstyle(paragraph, "BodyText")

    thesis_title_parts: list[str] = []
    collecting_title = False
    for paragraph in top_paragraphs:
        if paragraph not in cover_paragraphs:
            continue
        text = collapse_spaces(get_text(paragraph))
        if not text:
            continue
        if text.startswith("题"):
            collecting_title = True
            cleaned = re.sub(r"^题\s*目\s*", "", text)
            if cleaned:
                thesis_title_parts.append(cleaned)
            continue
        if collecting_title:
            if text.startswith(("学", "专", "学生姓名", "学 生 姓 名", "导师姓名", "导 师 姓 名", "班", "学号")):
                break
            thesis_title_parts.append(text)

    thesis_title = "".join(thesis_title_parts).replace(" ", "")
    if not thesis_title:
        thesis_title = "毕业设计论文"

    chapter_titles = [collapse_spaces(get_text(p)) for p in chapter_starts]
    header_specs = [
        ("header1.xml", "摘 要", False),
        ("header2.xml", "Abstract", False),
        ("header3.xml", thesis_title, True),
        ("header4.xml", chapter_titles[0], False),
        ("header5.xml", chapter_titles[1], False),
        ("header6.xml", chapter_titles[2], False),
        ("header7.xml", chapter_titles[3], False),
        ("header8.xml", chapter_titles[4], False),
        ("header9.xml", chapter_titles[5], False),
        ("header10.xml", chapter_titles[6], False),
        ("header11.xml", chapter_titles[7], False),
        ("header12.xml", "参考文献", False),
    ]

    for filename, header_title, page_on_left in header_specs:
        file_map[f"word/{filename}"] = create_header(header_title, page_on_left)

    header_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"
    for rel in list(rels_root):
        if rel.get("Type") == header_type or rel.get("Type", "").endswith("/footer"):
            rels_root.remove(rel)

    used_ids = set(rel.get("Id") for rel in rels_root if rel.get("Id"))
    next_rid = 200

    def new_rid() -> str:
        nonlocal next_rid
        while f"rId{next_rid}" in used_ids:
            next_rid += 1
        rid = f"rId{next_rid}"
        used_ids.add(rid)
        next_rid += 1
        return rid

    header_rids: dict[str, str] = {}
    for filename, _, _ in header_specs:
        rid = new_rid()
        header_rids[filename] = rid
        ET.SubElement(
            rels_root,
            qn("pr", "Relationship"),
            {
                "Id": rid,
                "Type": header_type,
                "Target": filename,
            },
        )

    boundary_config = [
        ("headerless", None, None, None, None),
        ("cn_abs", "header1.xml", "header1.xml", 1, "upperRoman"),
        ("en_abs", "header2.xml", "header2.xml", None, "upperRoman"),
        ("ch1", "header4.xml", "header3.xml", 1, None),
        ("ch2", "header5.xml", "header3.xml", None, None),
        ("ch3", "header6.xml", "header3.xml", None, None),
        ("ch4", "header7.xml", "header3.xml", None, None),
        ("ch5", "header8.xml", "header3.xml", None, None),
        ("ch6", "header9.xml", "header3.xml", None, None),
        ("ch7", "header10.xml", "header3.xml", None, None),
        ("ch8", "header11.xml", "header3.xml", None, None),
    ]

    for holder, (_, odd_file, even_file, start, fmt) in zip(boundary_holders, boundary_config):
        sectpr = build_sectpr(
            default_header_rid=header_rids.get(odd_file) if odd_file else None,
            even_header_rid=header_rids.get(even_file) if even_file else None,
            page_num_start=start,
            page_num_fmt=fmt,
            final_section=False,
        )
        add_or_replace_sectpr(holder, sectpr)

    body.append(
        build_sectpr(
            default_header_rid=header_rids["header12.xml"],
            even_header_rid=header_rids["header12.xml"],
            page_num_start=None,
            page_num_fmt=None,
            final_section=True,
        )
    )

    normalize_target_tables(body)
    cleanup_figure_artifacts(body)

    file_map["word/document.xml"] = ET.tostring(document_root, encoding="utf-8", xml_declaration=True)
    file_map["word/styles.xml"] = ET.tostring(styles_root, encoding="utf-8", xml_declaration=True)
    file_map["word/settings.xml"] = ET.tostring(settings_root, encoding="utf-8", xml_declaration=True)
    file_map["word/_rels/document.xml.rels"] = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)

    temp_path = target_path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in file_map.items():
            zf.writestr(name, data)

    temp_path.replace(target_path)


if __name__ == "__main__":
    main()
