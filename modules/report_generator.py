from fpdf import FPDF
import os

class PDFReport(FPDF):
    def __init__(self, title):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)
        self.set_title(title)

    def add_heading(self, text):
        self.set_font("Arial", "B", 16)
        self.cell(200, 10, txt=text, ln=True, align="C")
        self.ln(10)

    def add_paragraph(self, text):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, txt=text)
        self.ln()

    def add_image(self, image_path, w=180):
        if os.path.exists(image_path):
            self.image(image_path, w=w)
            self.ln()

def generate_pdf_report(title, metrics: dict, image_paths: list, output_path: str):
    pdf = PDFReport(title)
    pdf.add_heading("Model Evaluation Report")

    for key, value in metrics.items():
        if isinstance(value, float):
            pdf.add_paragraph(f"{key}: {value:.4f}")
        elif isinstance(value, str):
            pdf.add_paragraph(f"{key}: {value}")

    for img_path in image_paths:
        pdf.add_image(img_path)

    pdf.output(output_path)
    return output_path
