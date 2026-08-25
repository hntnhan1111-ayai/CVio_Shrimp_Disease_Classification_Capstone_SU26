from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "final_report_xelatex"
DATA = json.loads((REPORT / "source_data" / "chapters_1_2_semantic.json").read_text(encoding="utf-8"))
OUT = REPORT / "chapters_1_2_review.tex"


SPLIT_WORDS = {
    "An- droid": "Android", "Fourier- domain": "Fourier-domain", "Segmen- tation": "Segmentation",
    "SimAM- DCFR": "SimAM-DCFR", "Supervi- sor": "Supervisor", "Ultra- lytics": "Ultralytics",
    "ac- curacy": "accuracy", "an- notator": "annotator", "application- recorded": "application-recorded",
    "attention- based": "attention-based", "attention- placement": "attention-placement",
    "branch- aware": "branch-aware", "cam- era": "camera", "clas- sification": "classification",
    "clas- sifier": "classifier", "classification- oriented": "classification-oriented",
    "classifier- to": "classifier-to", "col- laboration": "collaboration", "col- lected": "collected",
    "com- parisons": "comparisons", "con- sistent": "consistent", "configu- ration": "configuration",
    "consoli- dation": "consolidation", "demon- stration": "demonstration", "deploy- ment": "deployment",
    "device- and": "device- and", "disease- region": "disease-region", "distur- bance": "disturbance",
    "du- plicate": "duplicate", "equiv- alent": "equivalent", "eval- uation": "evaluation",
    "evalua- tion": "evaluation", "evi- dence": "evidence", "ex- cluded": "excluded",
    "ex- periments": "experiments", "ex- plicitly": "explicitly", "exe- cuted": "executed",
    "experimen- tation": "experimentation", "exten- sion": "extension", "for- mally": "formally",
    "four- class": "four-class", "iden- tify": "identify", "interpre- tation": "interpretation",
    "interpreta- tion": "interpretation", "labora- tory": "laboratory", "leak- age": "leakage",
    "leakage- prone": "leakage-prone", "man- days": "man-days", "mask- accurate": "mask-accurate",
    "negative- image": "negative-image", "neural- network": "neural-network",
    "non- reproducible": "non-reproducible", "parti- tions": "partitions",
    "partition- wise": "partition-wise", "perfor- mance": "performance", "pre- diction": "prediction",
    "pre- dictions": "predictions", "pre- ventive": "preventive", "princi- pal": "principal",
    "pro- duces": "produces", "re- covered": "recovered", "re- search": "research",
    "re- view": "review", "rep- resentative": "representative", "robust- ness": "robustness",
    "seg- mentation": "segmentation", "segmen- tation": "segmentation", "segmenta- tion": "segmentation",
    "struc- ture": "structure", "su- pervisor": "supervisor", "three- camera": "three-camera",
    "thresh- old": "threshold", "un- supported": "unsupported", "valid- ity": "validity",
    "validation- selected": "validation-selected",
}


def clean(value: str) -> str:
    value = value.replace("\u037e", ";").replace("\u00ad", "").replace("–", "-").replace("—", "-")
    for old, new in SPLIT_WORDS.items():
        value = value.replace(old, new)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace("Project Scope & Limitations", "Project Scope and Limitations")
    value = value.replace("2 x NVIDIA", "2 × NVIDIA")
    value = value.replace("V erify", "Verify").replace("V ersion", "Version")
    value = value.replace("lightweighttrên", "lightweight trên").replace("thiết bịdi", "thiết bị di")
    value = value.replace("\u037e", ";")
    return value


def tex(value: str) -> str:
    value = clean(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "×": r"$\times$",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def block(chapter: str, page: int, index: int) -> dict:
    return DATA[chapter][page - 1]["blocks"][index]


def text(chapter: str, page: int, index: int) -> str:
    return clean(block(chapter, page, index)["text"])


def table(chapter: str, page: int, index: int = 0) -> list[list[str]]:
    return [[clean(cell) for cell in row] for row in DATA[chapter][page - 1]["tables"][index]["rows"]]


class Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def paragraph(self, value: str, revised: bool = False) -> None:
        value = tex(value)
        if revised:
            self.add(r"\begin{revisionblock}", value, r"\end{revisionblock}")
        else:
            self.add(value + r"\par")

    def source_paragraph(self, chapter: str, page: int, index: int, revised: bool | None = None) -> None:
        item = block(chapter, page, index)
        if revised is None:
            revised = item["highlight_ratio"] > 0.5
        self.paragraph(item["text"], revised)

    def heading(self, level: int, title: str, revised: bool = False) -> None:
        cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[level]
        rendered = tex(re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", clean(title)))
        if revised:
            self.add(rf"\{cmd}{{\texorpdfstring{{\colorbox{{RevisionYellow}}{{{rendered}}}}}{{{rendered}}}}}")
        else:
            self.add(rf"\{cmd}{{{rendered}}}")

    def figure(self, filename: str, caption: str, width: str = r"0.94\textwidth", height: str = r"0.62\textheight") -> None:
        self.add(
            r"\begin{figure}[H]", r"\centering",
            rf"\includegraphics[width={width},height={height},keepaspectratio]{{assets/chapters_1_2/{filename}}}",
            rf"\caption{{{tex(caption)}}}", r"\end{figure}",
        )

    def longtable(
        self,
        rows: list[list[str]],
        caption: str,
        widths: list[float],
        compact: bool = False,
        highlighted_rows: set[int] | None = None,
        parent_rows: set[int] | None = None,
    ) -> None:
        highlighted_rows = highlighted_rows or set()
        parent_rows = parent_rows or set()
        columns = "|" + "|".join(rf">{{\centering\arraybackslash}}p{{{w:.2f}cm}}" for w in widths) + "|"
        size = r"\scriptsize" if compact else r"\footnotesize"
        self.add(r"\begingroup", size, r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.16}")
        self.add(rf"\begin{{longtable}}{{{columns}}}")
        self.add(rf"\caption{{{tex(caption)}}}\\", r"\hline")
        header = " & ".join(rf"\textbf{{{tex(cell)}}}" for cell in rows[0]) + r" \\ \hline"
        self.add(r"\rowcolor{HeaderPeach}" + header, r"\endfirsthead")
        self.add(rf"\caption[]{{{tex(caption)} (continued)}}\\", r"\hline", r"\rowcolor{HeaderPeach}" + header, r"\endhead")
        self.add(r"\hline", rf"\multicolumn{{{len(widths)}}}{{r}}{{\footnotesize Continued on next page}}\\", r"\endfoot", r"\hline", r"\endlastfoot")
        for idx, row in enumerate(rows[1:], 1):
            if not any(cell.strip() for cell in row):
                continue
            cells = [tex(cell) for cell in row]
            if idx in parent_rows:
                cells = [rf"\textbf{{{cell}}}" for cell in cells]
            prefix = r"\rowcolor{RevisionYellow}" if idx in highlighted_rows else ""
            self.add(prefix + " & ".join(cells) + r" \\ \hline")
        self.add(r"\end{longtable}", r"\endgroup")


def merge_tables(parts: list[list[list[str]]]) -> list[list[str]]:
    result = [parts[0][0]]
    for part in parts:
        result.extend(row for row in part[1:] if any(cell.strip() for cell in row))
    return result


def build_wbs() -> list[list[str]]:
    parts = [table("chapter2", p) for p in (9, 10, 11, 12)]
    rows: list[list[str]] = [["#", "WBS Item", "Complexity", "Est. effort (man-days)"]]
    for part in parts:
        for raw in part[1:]:
            if not any(raw):
                continue
            if raw[0].startswith("Total"):
                continue
            if len(raw) == 5:
                raw = [raw[0], raw[1], raw[2] or raw[3], raw[4]]
            if not raw[0]:
                continue
            rows.append(raw[:4])
    missing = {
        "2.3": ["2.4", "Implement background-removal preprocessing", "Complex", "6"],
        "5.5": ["5.6", "Retain Healthy images as empty-label negatives", "Simple", "2"],
        "8.2": ["8.3", "Write project introduction and management plan", "Complex", "7"],
    }
    with_missing: list[list[str]] = []
    for row in rows:
        with_missing.append(row)
        if row[0] in missing:
            with_missing.append(missing[row[0]])
    with_missing.append(["", "Total Estimated Effort (man-days)", "", "487"])
    return with_missing


def preamble(w: Writer) -> None:
    w.add(r"\documentclass[12pt,a4paper]{report}")
    w.add(r"\usepackage{fontspec}", r"\setmainfont{Times New Roman}")
    w.add(r"\usepackage{geometry}", r"\geometry{left=2.15cm,right=2.15cm,top=1.6cm,bottom=1.8cm}")
    w.add(r"\usepackage{xcolor,colortbl,graphicx,float,longtable,array,ragged2e,caption,titlesec,fancyhdr,enumitem,microtype,hyperref,framed}")
    w.add(r"\definecolor{RevisionYellow}{RGB}{255,255,0}", r"\definecolor{HeaderPeach}{RGB}{249,218,199}", r"\definecolor{ChapterRed}{RGB}{193,0,0}")
    w.add(r"\colorlet{shadecolor}{RevisionYellow}", r"\newenvironment{revisionblock}{\begin{snugshade}\noindent}{\end{snugshade}}")
    w.add(r"\renewcommand{\thechapter}{\Roman{chapter}}", r"\renewcommand{\thesection}{\arabic{section}}", r"\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}", r"\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}")
    w.add(r"\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}", r"\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}")
    w.add(r"\setcounter{secnumdepth}{3}", r"\setlength{\emergencystretch}{2em}")
    w.add(r"\titleformat{\chapter}[hang]{\bfseries\color{ChapterRed}\fontsize{18}{22}\selectfont}{\thechapter.}{0.55em}{}", r"\titlespacing*{\chapter}{0pt}{0pt}{12pt}")
    w.add(r"\titleformat{\section}{\bfseries\fontsize{15}{18}\selectfont}{\thesection.}{0.5em}{}", r"\titleformat{\subsection}{\bfseries\fontsize{13.5}{16}\selectfont}{\thesubsection}{0.5em}{}", r"\titleformat{\subsubsection}{\bfseries\fontsize{12}{14}\selectfont}{\thesubsubsection}{0.5em}{}")
    w.add(r"\titlespacing*{\section}{0pt}{11pt}{5pt}", r"\titlespacing*{\subsection}{0pt}{9pt}{4pt}", r"\titlespacing*{\subsubsection}{0pt}{7pt}{3pt}")
    w.add(r"\captionsetup{font={small},labelfont=bf,labelsep=period,justification=centering,singlelinecheck=false}")
    w.add(r"\setlength{\parindent}{1.25cm}", r"\setlength{\parskip}{3pt}", r"\linespread{1.18}")
    w.add(r"\setlist{leftmargin=1.1cm,itemsep=4pt,topsep=4pt}")
    w.add(r"\pagestyle{fancy}", r"\fancyhf{}", r"\fancyfoot[C]{\thepage}", r"\renewcommand{\headrulewidth}{0pt}")
    w.add(r"\hypersetup{hidelinks}")
    w.add(r"\begin{document}", r"\pagenumbering{arabic}")


def chapter_one(w: Writer) -> None:
    w.add(r"\chapter{Project Introduction}")
    w.heading(1, text("chapter1", 1, 1))
    w.heading(2, text("chapter1", 1, 2))
    w.source_paragraph("chapter1", 1, 3)
    w.longtable(table("chapter1", 1), "Project information", [4.1, 11.7], highlighted_rows=set(range(1, 13)))
    w.source_paragraph("chapter1", 1, 5)
    w.longtable(table("chapter1", 2, 0), "Supervisor information", [4.3, 5.1, 2.8, 3.6])
    w.longtable(table("chapter1", 2, 1), "Team member information", [4.2, 5.0, 2.8, 3.8])
    w.source_paragraph("chapter1", 2, 2)
    w.heading(2, text("chapter1", 2, 3), revised=True)
    for idx in (4, 5, 6):
        w.source_paragraph("chapter1", 2, idx)
    w.paragraph(text("chapter1", 2, 7) + " " + text("chapter1", 3, 0), True)
    w.source_paragraph("chapter1", 3, 1)
    w.source_paragraph("chapter1", 3, 2)
    w.figure("figure_1_1_scope_evolution.png", "Evolution of the project scope.")
    w.source_paragraph("chapter1", 3, 4)
    w.source_paragraph("chapter1", 3, 5)
    w.source_paragraph("chapter1", 4, 0)
    w.figure("figure_1_2_technology_stack.png", "Project technology stack.")
    w.source_paragraph("chapter1", 4, 2)
    w.heading(1, text("chapter1", 4, 3), revised=True)
    w.paragraph(text("chapter1", 4, 4) + " " + text("chapter1", 5, 0), True)
    w.source_paragraph("chapter1", 5, 1)
    w.source_paragraph("chapter1", 5, 2)
    w.heading(1, text("chapter1", 5, 3), revised=True)
    w.source_paragraph("chapter1", 5, 4)
    objectives = [text("chapter1", 5, i) for i in range(5, 13)] + [text("chapter1", 6, i) for i in range(2)]
    w.add(r"\begin{enumerate}")
    for item in objectives:
        item = re.sub(r"^\d+\.\s*", "", item)
        w.add(r"\item " + (r"\colorbox{RevisionYellow}{\parbox{0.91\linewidth}{" + tex(item) + "}}" if "YOLO" in item or "specimen" in item or "Android" in item else tex(item)))
    w.add(r"\end{enumerate}")
    w.heading(1, text("chapter1", 6, 2), revised=True)
    for idx in (3, 4, 5):
        w.source_paragraph("chapter1", 6, idx)
    w.heading(1, text("chapter1", 6, 6), revised=True)
    for idx in (7, 8):
        w.source_paragraph("chapter1", 6, idx)
    w.heading(1, text("chapter1", 6, 9), revised=True)
    w.paragraph(text("chapter1", 6, 10) + " " + text("chapter1", 7, 0), True)
    w.heading(2, text("chapter1", 7, 1), revised=True)
    for idx in (2, 3, 4):
        w.source_paragraph("chapter1", 7, idx)
    scope = merge_tables([table("chapter1", 7), table("chapter1", 8)])
    w.longtable(scope, "Final project scope and explicit limitations", [3.1, 6.0, 6.7], highlighted_rows=set(range(1, len(scope))))
    w.source_paragraph("chapter1", 8, 1)
    w.heading(2, text("chapter1", 8, 2), revised=True)
    w.source_paragraph("chapter1", 8, 3)
    w.paragraph(text("chapter1", 8, 4) + " " + text("chapter1", 8, 5), True)
    w.source_paragraph("chapter1", 8, 6)
    w.paragraph(text("chapter1", 8, 7) + " " + text("chapter1", 9, 0), True)
    w.source_paragraph("chapter1", 9, 1)


def chapter_two(w: Writer) -> None:
    w.add(r"\chapter{Project Management Plan}")
    w.paragraph(text("chapter2", 1, 7) + " " + text("chapter2", 1, 1))
    w.source_paragraph("chapter2", 1, 2)
    w.heading(1, text("chapter2", 1, 3))
    w.heading(2, text("chapter2", 1, 4))
    w.source_paragraph("chapter2", 1, 5)
    team = merge_tables([table("chapter2", 1), table("chapter2", 2)])
    w.longtable(team, "Team structure and roles", [3.0, 2.2, 2.7, 7.6])
    w.source_paragraph("chapter2", 2, 0)
    w.source_paragraph("chapter2", 2, 1)
    w.figure("figure_2_1_team_structure.jpeg", "Team structure and research responsibilities.")
    w.source_paragraph("chapter2", 2, 3)
    w.longtable(table("chapter2", 3), "Roles and responsibilities", [4.0, 11.8])
    w.heading(3, text("chapter2", 3, 1))
    w.source_paragraph("chapter2", 3, 2)
    w.source_paragraph("chapter2", 3, 3)
    w.heading(3, text("chapter2", 3, 4))
    w.paragraph("Role allocation changed when the research scope expanded. During the initial classification phase, Tran Huu Nhan identified the source dataset and Nguyen Van Phong led the first dataset inspection, preprocessing, and baseline-model benchmarking. After the first review and the formal addition of instance segmentation around Session 3, the work was separated more sharply. Nhan concentrated on YOLO26m-cls selection, ASL-LDAM, SimAM-DCFR, classification robustness, XAI, external classification experiments, and integration. Nhu combined Android/Firebase development with attention-based segmentation research. Phong took ownership of mask semantics, single-annotator labeling, grouped splitting, leakage analysis, segmentation baselines, and Fourier-domain experiments.")
    w.source_paragraph("chapter2", 4, 0)
    w.heading(3, text("chapter2", 4, 1))
    w.source_paragraph("chapter2", 4, 2)
    w.source_paragraph("chapter2", 4, 3)
    w.heading(2, text("chapter2", 4, 4))
    w.source_paragraph("chapter2", 4, 5)
    communication = merge_tables([table("chapter2", 4), table("chapter2", 5)])
    communication[1][1] = communication[1][1].rstrip() + " assign next tasks"
    w.longtable(communication, "Project communication plan", [3.6, 7.2, 2.1, 2.9])
    w.source_paragraph("chapter2", 5, 0)
    w.heading(3, text("chapter2", 5, 1))
    w.paragraph("The weekly Google Meet is the formal decision point of the project. Each meeting begins with the results and unresolved issues from the previous task cycle. Members report what was attempted, which artifacts were produced, whether the expected acceptance condition was met, and which technical or scheduling constraints remain. The supervisor then provides direction and assigns the next work, while the project leader records ownership, dependencies, and the expected deadline. In most weeks, the deadline is placed before the following meeting so that the next discussion is based on current evidence rather than unverified progress statements.")
    w.source_paragraph("chapter2", 6, 1)
    w.heading(3, text("chapter2", 6, 2))
    w.source_paragraph("chapter2", 6, 3)
    w.source_paragraph("chapter2", 6, 4)
    w.heading(3, text("chapter2", 6, 5))
    w.source_paragraph("chapter2", 6, 6)
    w.source_paragraph("chapter2", 6, 7)
    w.heading(1, text("chapter2", 6, 8))
    w.paragraph(text("chapter2", 6, 9) + " and evidence.")
    w.paragraph("Figure 2.2 maps the original six CRISP-DM phases to the project. Dataset and experimental evidence are placed at the center because data quality, split validity, labeling, and traceable experiment outputs influence every phase.")
    w.figure("figure_2_2_crisp_dm.jpeg", "CRISP-DM applied to the project.")
    w.source_paragraph("chapter2", 7, 2)
    w.longtable(table("chapter2", 7), "Application of CRISP-DM phases to the project", [4.0, 11.8])
    w.source_paragraph("chapter2", 8, 0)
    w.figure("figure_2_3_weekly_cycle.jpeg", "Weekly research execution cycle.")
    w.source_paragraph("chapter2", 8, 2)
    w.paragraph("CRISP-DM was selected because it provides a recognized lifecycle for projects in which data characteristics, preparation choices, model behavior, evaluation findings, and deployment constraints repeatedly influence one another. The framework is more suitable than a formal Scrum description for the project because weekly work is driven by research questions and experimental outcomes rather than by a fixed software increment and a complete test suite. The six CRISP-DM phase names are retained without inventing a new methodology name.")
    w.paragraph("Business Understanding establishes the initial four-class classification, robustness, lightweight deployment, intended users, and non-clinical boundary. Data Understanding investigates source datasets, class distribution, image variability, repeated specimens, and annotation feasibility. Data Preparation is task-specific: classification uses background removal and fixed image-level partitions, whereas segmentation requires team-created disease-region masks, Healthy empty-label negatives, and later a grouped-specimen split. These differences prevent the project from applying one preprocessing assumption indiscriminately to both tasks.")
    w.paragraph("Modeling includes controlled screening of TIMM and official Ultralytics classification models, selection of YOLO26m-cls under the performance and approximate parameter constraint, implementation of ASL-LDAM and SimAM-DCFR, and the segmentation attention and Fourier studies. Evaluation checks clean performance, corruption behavior, classwise outcomes, qualitative XAI, mask metrics, leakage effects, and failure cases. Deployment converts selected outputs into Android, TensorFlow Lite, Firebase, farmer, and administrator workflows. Findings in any later phase may require revision of an earlier phase.")
    w.source_paragraph("chapter2", 9, 0)
    w.heading(2, text("chapter2", 9, 1))
    w.source_paragraph("chapter2", 9, 2)
    w.source_paragraph("chapter2", 9, 4)
    w.source_paragraph("chapter2", 9, 5)
    wbs = build_wbs()
    parent_rows = {i for i, row in enumerate(wbs[1:], 1) if row[0].isdigit() or row[0] == ""}
    highlighted = {i for i, row in enumerate(wbs[1:], 1) if any(key in " ".join(row) for key in ("Healthy", "grouped", "Fourier", "three-device", "two-model", "initial defense"))}
    w.longtable(wbs, "Detailed Work Breakdown Structure and estimated effort", [1.2, 9.3, 2.6, 2.4], compact=True, highlighted_rows=highlighted, parent_rows=parent_rows)
    w.source_paragraph("chapter2", 12, 0)
    w.figure("figure_2_4_gantt.jpeg", "Project Gantt chart.", width=r"0.98\textwidth", height=r"0.70\textheight")
    w.source_paragraph("chapter2", 12, 2)
    w.source_paragraph("chapter2", 15, 5)
    w.longtable(table("chapter2", 13), "Responsibility assignments", [4.5, 2.6, 2.6, 2.6, 2.6], compact=True)
    w.source_paragraph("chapter2", 13, 0)
    for p, idx in [(13, 2), (13, 3), (13, 4), (14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (15, 0), (15, 1)]:
        value = text("chapter2", p, idx)
        if re.match(r"^2\.1\.\d", value):
            w.heading(3, value)
        else:
            w.source_paragraph("chapter2", p, idx)
    w.heading(2, text("chapter2", 15, 2))
    w.source_paragraph("chapter2", 15, 3)
    risks = merge_tables([table("chapter2", 15), table("chapter2", 16)])
    risks[4][3] = "Fix dataset split, seed, preprocessing, checkpoint selection, metrics, class order, and evaluation scripts for direct comparisons."
    w.longtable(risks, "Project risks and response plans", [1.0, 4.8, 3.4, 6.2], compact=True, highlighted_rows={1, 2, 4, 10, 11, 12})
    w.source_paragraph("chapter2", 16, 0)
    for idx in range(0, 10):
        value = text("chapter2", 17, idx)
        if re.match(r"^2\.2\.\d", value):
            w.heading(3, value)
        else:
            w.source_paragraph("chapter2", 17, idx)
    w.paragraph(text("chapter2", 17, 9) + " validates TensorFlow Lite outputs against the source framework before packaging, preserves model metadata, and avoids converting application-recorded elapsed time into an unsupported real-time claim. Firebase and network failures are separated from local inference so that service interruptions do not redefine the model result.")
    w.heading(3, text("chapter2", 18, 1))
    w.source_paragraph("chapter2", 18, 2)
    w.source_paragraph("chapter2", 18, 3)
    w.heading(2, "2.3 Quality Management")
    w.source_paragraph("chapter2", 18, 4)
    quality = merge_tables([table("chapter2", 18), table("chapter2", 19)])
    w.longtable(quality, "Quality management plan", [3.2, 8.1, 4.5], highlighted_rows={1, 3, 4, 5, 7, 8})
    w.source_paragraph("chapter2", 19, 0)
    w.figure("figure_2_5_artifact_flow.png", "Document and experiment artifact flow.")
    w.source_paragraph("chapter2", 20, 1)
    w.heading(3, text("chapter2", 20, 2))
    w.source_paragraph("chapter2", 20, 3)
    w.source_paragraph("chapter2", 20, 4)
    w.heading(3, "2.3.2 Experimental and Model-Comparison Quality")
    w.paragraph(text("chapter2", 20, 5) + " interpretation, but they do not replace the declared primary selection rule.")
    w.source_paragraph("chapter2", 21, 8)
    w.source_paragraph("chapter2", 21, 0)
    w.heading(3, text("chapter2", 21, 1))
    w.source_paragraph("chapter2", 21, 2)
    w.source_paragraph("chapter2", 21, 3)
    w.heading(3, text("chapter2", 21, 4))
    w.source_paragraph("chapter2", 21, 9)
    w.source_paragraph("chapter2", 21, 5)
    w.heading(2, text("chapter2", 21, 6))
    w.paragraph("No direct project budget was incurred. The team used public datasets, open-source frameworks, free Kaggle Notebook GPU allocations, existing Android devices, Firebase free-tier services, institutional resources, and no-cost collaboration tools. The table documents the resource model without inventing monetary expenditure.")
    w.longtable(table("chapter2", 22), "Project resources and funding status", [3.7, 7.7, 4.4])
    w.heading(3, text("chapter2", 22, 1))
    w.source_paragraph("chapter2", 22, 2)
    w.source_paragraph("chapter2", 22, 3)
    w.heading(3, text("chapter2", 22, 4))
    w.source_paragraph("chapter2", 22, 5)
    w.paragraph("Resource changes are reviewed when they affect scientific comparability or delivery. Moving from one GPU environment to another, changing image size to fit memory, reducing epoch count, or replacing a mobile model format may alter the result and therefore cannot be treated only as a budget decision. The effect on protocol, schedule, and thesis interpretation must be assessed through change management.")
    w.heading(3, text("chapter2", 23, 1))
    w.source_paragraph("chapter2", 23, 2)
    w.source_paragraph("chapter2", 23, 3)
    w.heading(2, text("chapter2", 23, 4))
    w.source_paragraph("chapter2", 23, 5)
    w.source_paragraph("chapter2", 23, 6)
    w.figure("figure_2_6_change_process.jpeg", "Controlled change-management process.")
    w.longtable(table("chapter2", 24), "Change categories and control requirements", [4.0, 5.7, 6.1])
    w.heading(3, text("chapter2", 24, 1))
    w.source_paragraph("chapter2", 24, 2)
    w.source_paragraph("chapter2", 24, 3)
    w.heading(3, text("chapter2", 24, 4))
    w.paragraph("Impact assessment examines scope, owner workload, dataset requirements, implementation effort, GPU time, dependencies, mobile effects, documentation changes, and the validity of previously produced evidence. The team determines whether earlier work remains usable, must be rerun, or needs to be relabeled. A proposed change is accepted only when its value and required work are understood; otherwise it is revised, postponed, or rejected.")
    w.source_paragraph("chapter2", 24, 6)
    w.source_paragraph("chapter2", 25, 0)
    w.heading(3, text("chapter2", 25, 1))
    w.source_paragraph("chapter2", 25, 2)
    w.source_paragraph("chapter2", 25, 3)
    w.heading(3, text("chapter2", 25, 4))
    w.source_paragraph("chapter2", 25, 5)
    w.source_paragraph("chapter2", 25, 6)
    w.heading(3, text("chapter2", 25, 7), revised=True)
    w.paragraph(text("chapter2", 25, 8) + " evidence, and response-letter traceability.", revised=True)
    w.heading(2, text("chapter2", 26, 0), revised=True)
    w.source_paragraph("chapter2", 26, 1)
    closure_rows = [
        ["Closure criterion", "Definition of Done"],
        ["Classification research complete", "The final proposed classifier is fully specified; Macro-F1 is reported consistently as 91.01%; clean and corruption comparisons are traceable; component-evidence wording is internally consistent."],
        ["Segmentation research complete", "BG/WSSV labels, Healthy empty-label handling, the specimen-grouped 1,149-image subset, the unmatched-name expansion boundary, baselines, attention/Fourier evidence, metrics, qualitative analysis, and limitations are documented."],
        ["Improvement criterion satisfied", "Each claimed proposed method is compared against the relevant controlled baseline and is supported by the declared metric and evidence."],
        ["Research novelty documented", "The thesis clearly identifies what is adopted, reimplemented, modified, or newly proposed; unsupported novelty claims are removed."],
        ["Reproducibility package complete", "Datasets, split rules, seed, environment, configurations, code, checkpoints, logs, metrics, figures, and model metadata are archived."],
        ["Mobile prototype complete", "One common Android application/model configuration executes the FP32 classifier for every accepted image and conditionally invokes the FP16 segmenter; branch-aware operational runtime from three Android devices is archived without claiming field accuracy or iOS support."],
        ["Paper and thesis complete", "Classification paper material, segmentation documentation, tables, figures, references, limitations, and conclusions are integrated and reviewed."],
        ["Final quality audit complete", "Section numbering, A4 page geometry, Times New Roman typography, table/figure placement, yellow revision highlighting, terminology, metrics, dates, limitations, and cross-document consistency pass the final rendered audit."],
        ["Defense preparation complete", "Presentation content, demonstration path, role allocation, likely questions, and evidence references are prepared."],
        ["Final submission complete", "The 5 August initial defense package and the Council-mandated revised package due before 12:00 on 15 August 2026 are both traceable; accepted artifacts are archived."],
    ]
    w.longtable(closure_rows, "Project closure and Definition of Done", [4.4, 11.4], highlighted_rows={2, 6, 8, 10})
    w.source_paragraph("chapter2", 26, 28)
    for page, indices in [(27, range(0, 8)), (28, range(0, 5))]:
        for idx in indices:
            value = text("chapter2", page, idx)
            if re.match(r"^2\.6\.\d", value):
                w.heading(3, value, revised=True)
            else:
                w.source_paragraph("chapter2", page, idx)


def main() -> None:
    w = Writer()
    preamble(w)
    chapter_one(w)
    chapter_two(w)
    w.add(r"\end{document}")
    OUT.write_text("\n".join(line for line in w.lines if line != "") + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
