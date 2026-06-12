import logging
import os
from pathlib import Path

from cv2.typing import MatLike

from ..layout import LayoutRunner as LayoutEngine
from ..layout.backends.base import Backend


class LayoutRunner:
    def __init__(self) -> None:
        self.logger = logging.getLogger("layout_runner")

        model_dir = Path(os.getenv("LAYOUT_MODEL_DIR", "./layout_model"))
        backend = Backend[os.getenv("RUNTIME_BACKEND", "ONNX")]
        threads = int(os.getenv("BACKEND_THREADS", "2"))
        self.model_list = [
            m.strip() for m in os.getenv("MODEL_LIST", "25").split(",") if m.strip()
        ]

        configs = [
            {
                "model_id": model_id,
                "backend": backend,
                "model_path": model_dir / f"RT-DETR_{model_id}.{backend.value}",
                "threads": threads,
            }
            for model_id in self.model_list
        ]
        self.logger.info("Loading layout models: %s (backend=%s)", configs, backend)
        self.engine = LayoutEngine(configs)

    def _apply_model_9_rules(
        self,
        bboxes: list[list[float]],
        model_id: str,
        aspect_ratio_threshold: float = 0.2,
    ) -> None:
        """
        仅针对 model_id 为 9：若类别为 text 且 bbox 宽高比(宽/高)小于阈值，则改为 supplement。
        bboxes 每行为 [x1, y1, x2, y2, score, cls]，原地修改 cls。
        """
        if model_id != "9":
            return
        text_cls = self.classes_9.index("text")
        supplement_cls = self.classes_9.index("supplement")
        for b in bboxes:
            if len(b) < 6:
                continue
            if int(b[5]) != text_cls:
                continue
            w = abs(b[2] - b[0])
            h = abs(b[3] - b[1])
            if h <= 0:
                continue
            if w / h < aspect_ratio_threshold:
                b[5] = float(supplement_cls)

    async def inference(self, image: MatLike, model_id: str) -> list[list[float]]:
        result: list[list[float]] = await self.engine[model_id].inference(image)
        self._apply_model_9_rules(result, model_id)
        # self.logger.info(f"Layout result: {result}")
        return result

    async def inference_gather(
        self, image: MatLike, model_id: str
    ) -> list[dict[str, list[float] | float | str | tuple[int, ...]]]:
        bboxes = await self.inference(image, model_id)
        results = list[dict[str, list[float] | float | str | tuple[int, ...]]]()
        for x1, y1, x2, y2, score, cls in bboxes:
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "cls": int(cls),
                    "label": self.classes_all[model_id][int(cls)],
                    "color": self.colors[int(cls)],
                }
            )
        return results


    def determine_columns(self, bboxes: list[list[float]], line1_splitting: float= 0.33, line2_splitting: float =0.60) -> int:
        """
        Determine the number of columns based on the position and width of the boxes.
        """

        if not bboxes or len(bboxes)<3:
            return 1

        # filter thin boxes like line_no, mnote etc. the width of the box should be larger than 0.1
        # consider that the left  and right  margines are larger than 0.05
        # these magic numbers are based on case study and may need to be tuned
        column1_total_height = sum([abs(box[3]-box[1]) for box in bboxes if box[0]<line1_splitting and box[2] - box[0] > 0.1])
        column2_total_height = sum([abs(box[3]-box[1]) for box in bboxes if box[0]>line1_splitting and box[0]<line2_splitting and box[2] - box[0] > 0.1])
        column3_total_height= sum([abs(box[3]-box[1]) for box in bboxes if box[0]>line2_splitting and  box[2] - box[0] > 0.1])
        if column3_total_height > min(0.2,column1_total_height*0.6):
            return 3
        if column2_total_height > min(0.2,column1_total_height*0.6):
            return 2
        return 1


    def column_sort(self, bboxes: list[list[float]]) -> list[list[float]]:
        """
        Sort the boxes in each column based on their coordinates.
        those boxes that span more than two columns are considered,for example, title line.
        """
        
        bboxes.sort(key=lambda b: b[1])  # global sort by y1
        column_num = self.determine_columns(bboxes)
        # print(column_num)
        # with open("column_num.txt", "a") as f:
        #     f.write(f"{column_num}\n")

        # 1 column just return
        if column_num == 1:
            return bboxes
        
        # have to consider titles or sections that locate in the center
        # so we need to sort bboxes above the center line
        col_1 = list[list[float]]()
        col_2 = list[list[float]]()
        col_3 = list[list[float]]()
        if column_num == 2:
            for b in bboxes:
                if b[0] < 0.4:  # first column or title line
                    if b[2]< 0.52: # first column
                        col_1.append(b) 
                    else:
                        col_2.append(b) # title line
                        # merge the boxes above the title line
                        col_1.extend(col_2)
                        col_2 = list[list[float]]()
                else:  # second column
                    col_2.append(b)
            return col_1 + col_2
        
        else:
            for b in bboxes:
                if b[0] < 0.3:  # first column or title line
                    if b[2] < 0.4:
                        col_1.append(b) # first column
                    elif b[2] < 0.66:   # across the first two columns
                        col_2.append(b)
                        col_1.extend(col_2)
                        col_2 = list[list[float]]()

                    else: # across three columns

                        col_3.append(b)
                        col_1.extend(col_2)
                        col_1.extend(col_3)
                        col_2 = list[list[float]]()
                        col_3 = list[list[float]]()
                    
                elif b[0] < 0.6: # second column or title line
                    if b[2]>0.66: # across the last two columns
                        col_3.append(b)
                        col_2.extend(col_3)
                        col_3 = list[list[float]]()

                    else: # the second column
                        col_2.append(b)
                else:  # the third column
                    col_3.append(b)

            return col_1 + col_2 + col_3
    

    def filter(
        self, bboxes: list[list[float]], options: dict[str, bool], model_id: str
    ) -> tuple[list[list[float]], list[str], list[str]]:
        bboxes = self.column_sort(bboxes)
        _bboxes = map(lambda b: (b[:5], self.classes_all[model_id][round(b[5])]), bboxes)

        target = list[str]()
        for category in options:
            if options[category] and category in self.classes_category[model_id].keys():
                target.extend(self.classes_category[model_id][category])

        filtered_bboxes = list(filter(lambda b: b[1] in target, _bboxes))

        if not filtered_bboxes:
            return list(), list(), list()

        b, c = zip(*filtered_bboxes)
        return list(b), list(c), list(map(lambda _c: self.classes_category_lut[model_id][_c], c))
     
    classes_chem = [
        "chem",
        "rxn",
    ]

    classes_5 = [
        "text",
        "tab",
        "fig",
        "eqn",
        "chem",
    ]

    classes_9 = [
        "text",
        "supplement",
        "noise",
        "tab",
        "graph",
        "fig",
        "eqn",
        "chem",
        "rxn",
        ]
    classes_25=[
        'title',
        'sec',
        'text',
        'photo',
        'seq',
        'head',
        'foot',
        'draw',
        'mnote',
        'cap',
        'struc',
        'figno',
        'lineno',
        'colno',
        'ref',
        'toc',
        'noise',
        'tab',
        'eqn',
        'chem',
        'figcx',
        'rxn',
        'bib',
        'srep',
        'graph'
        ]

    classes_all = {
        "5": classes_5,
        "9": classes_9,
        "25": classes_25,
        "chem": classes_chem,
    } 

    classes_category_chem: dict[str, list[str]] = {
        "chemical": ["chem", "rxn"],
    }

    classes_category_9: dict[str, list[str]] = {
        "figure": ["fig", "graph"],  # for backward compatibility
        "chemical": ["chem", "rxn"],
        "equation": ["eqn"],
        "table": ["tab"],
        "main_text": ["text"],
        "supplemental_text": ["supplement"],
        "others": ["noise"],
        # "supplemental_text": ["srep"],
        # "complex": ["srep"],
        # "others": ["noise", "mnote"],
    }

    classes_category_25: dict[str, list[str]] = {
        "figure": ["draw", "graph", "photo", "struc"],
        "chemical": ["chem", "rxn"],
        "equation": ["eqn"],
        "table": ["tab"],
        "main_text": ["text","cap", "figno", "sec", "seq", "text", "title", "ref", "toc"],
        "supplemental_text": ["supplement","colno", "foot", "head", "lineno", "mnote"],
        "complex": ["bib", "figcx", "srep"],
        "others": ["noise"],
    }
    classes_category: dict[str, dict[str, list[str]]] = {
        "5": classes_category_9,
        "9": classes_category_9,
        "25": classes_category_25,
        "chem": classes_category_chem,
    }

    classes_category_chem_lut: dict[str, str] = {
        "chem": "chemical",
        "rxn": "chemical",
    }

    classes_category_9_lut: dict[str, str] = {}
    for _category, _classes in classes_category_9.items():
        for _c in _classes:
            classes_category_9_lut[_c] = _category
    
    classes_category_25_lut: dict[str, str] = {}
    for _category, _classes in classes_category_25.items():
        for _c in _classes:
            classes_category_25_lut[_c] = _category
    
    classes_category_lut: dict[str, str] = {
        "9": classes_category_9_lut,
        "5": classes_category_9_lut,
        "25": classes_category_25_lut,
        "chem": classes_category_chem_lut,
    }

    colors = [
        (97, 242, 211),
        (13, 158, 56),
        (0, 139, 173),
        (13, 158, 56),
        (0, 139, 173),
        (211, 219, 92),
        (217, 109, 9),
        (13, 56, 212),
        (222, 84, 146),
        (171, 89, 247),
        (0, 139, 173),
        (158, 163, 255),
        (13, 56, 212),
        (105, 192, 255),
        (13, 56, 212),
        (158, 163, 255),
        (105, 192, 255),
        (211, 219, 92),
        (217, 109, 9),
        (255, 198, 173),
        (97, 242, 211),
        (13, 158, 56),
        (211, 219, 92),
        (217, 109, 9),
        (158, 163, 255)
        ]
