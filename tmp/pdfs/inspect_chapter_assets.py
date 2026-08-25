from pathlib import Path
import fitz

root = Path(r"C:\Users\Admin\workspace\CVio_Shrimp_Disease_Classification_Capstone_SU26")
src = root / "CVio_Codex_Final_Report_Handoff_2026-08-11" / "CVio_Final_Report_Split_2026-08-13" / "CVio_Final_Report_Split_2026-08-13"
for filename in ["01_chapter_I_project_introduction.pdf", "02_chapter_II_project_management_plan.pdf"]:
    doc = fitz.open(src / filename)
    print("\n", filename, "pages", len(doc))
    for pno, page in enumerate(doc, 1):
        images = page.get_image_info(xrefs=True)
        annots = list(page.annots() or [])
        tables = page.find_tables().tables
        drawings = page.get_drawings()
        if images or annots or tables:
            print("page", pno, "images", len(images), "annots", [(a.type, a.rect) for a in annots], "tables", len(tables), "drawings", len(drawings))
            for i, image in enumerate(images, 1):
                print(" image", i, {k: image.get(k) for k in ("xref", "width", "height", "bbox", "colorspace", "bpc")})
            for i, table in enumerate(tables, 1):
                print(" table", i, "bbox", table.bbox, "rows", table.row_count, "cols", table.col_count)
