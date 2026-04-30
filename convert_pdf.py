import sys
from markdown_pdf import MarkdownPdf, Section

def convert_to_pdf(input_file):
    print(f"Iniciando conversión de {input_file} a PDF...")
    pdf = MarkdownPdf()
    
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    pdf.add_section(Section(text))
    output_file = input_file.replace('.md', '.pdf')
    pdf.save(output_file)
    print(f"Conversión exitosa! Guardado como {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_to_pdf(sys.argv[1])
    else:
        print("Por favor, provee el nombre del archivo. Ejemplo: python convert_pdf.py archivo.md")

