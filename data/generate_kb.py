import json
import os
from fpdf import FPDF

kb_data = [
    {"subject": "the Python Programming Language", "filename": "doc_01_python.pdf", "fact": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.", "question": "Who created the Python programming language?", "answer": "Python was created by Guido van Rossum and first released in 1991."},
    {"subject": "the Speed of Light", "filename": "doc_02_light.pdf", "fact": "The speed of light in a vacuum is exactly 299,792,458 meters per second.", "question": "What is the exact speed of light in a vacuum?", "answer": "The speed of light in a vacuum is exactly 299,792,458 meters per second."},
    {"subject": "the Apollo 11 Mission", "filename": "doc_03_apollo.pdf", "fact": "Apollo 11 was the historic American spaceflight that first landed humans on the Moon on July 20, 1969.", "question": "Which spaceflight was the first to land humans on the Moon?", "answer": "Apollo 11 was the spaceflight that first landed humans on the Moon on July 20, 1969."},
    {"subject": "Cellular Biology", "filename": "doc_04_biology.pdf", "fact": "The mitochondria are membrane-bound cell organelles widely known as the powerhouse of the cell.", "question": "What organelle is known as the powerhouse of the cell?", "answer": "The mitochondria are widely known as the powerhouse of the cell."},
    {"subject": "Parisian Architecture", "filename": "doc_05_paris.pdf", "fact": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, completed in 1889.", "question": "In what year was the Eiffel Tower completed?", "answer": "The Eiffel Tower was completed in 1889."},
    {"subject": "Water Chemistry", "filename": "doc_06_water.pdf", "fact": "Water is a vital chemical substance composed of two hydrogen atoms covalently bonded to one oxygen atom (H2O).", "question": "What is the chemical composition of water?", "answer": "Water is composed of two hydrogen atoms covalently bonded to one oxygen atom (H2O)."},
    {"subject": "Ancient Fortifications", "filename": "doc_07_china.pdf", "fact": "The Great Wall of China is a massive series of fortifications built across the historical northern borders of ancient Chinese states.", "question": "Why was the Great Wall of China built?", "answer": "It is a series of fortifications built across the historical northern borders of ancient Chinese states."},
    {"subject": "JavaScript", "filename": "doc_08_javascript.pdf", "fact": "JavaScript is a versatile, just-in-time compiled programming language that conforms to the ECMAScript standard.", "question": "What standard does the JavaScript programming language conform to?", "answer": "JavaScript conforms to the ECMAScript standard."},
    {"subject": "Theoretical Physics", "filename": "doc_09_physics.pdf", "fact": "Albert Einstein developed the theory of relativity, which remains one of the two foundational pillars of modern physics.", "question": "Who developed the theory of relativity?", "answer": "The theory of relativity was developed by Albert Einstein."},
    {"subject": "Oceanography", "filename": "doc_10_ocean.pdf", "fact": "The Mariana Trench is the deepest oceanic trench on Earth, located in the western Pacific Ocean.", "question": "Where is the deepest oceanic trench on Earth located?", "answer": "The Mariana Trench is located in the western Pacific Ocean."},
    {"subject": "Astrophysics", "filename": "doc_11_space.pdf", "fact": "A black hole is a region of spacetime where the gravitational pull is so intense that nothing, not even light, can escape it.", "question": "Why can't light escape from a black hole?", "answer": "Light cannot escape because the gravitational pull in that region of spacetime is too intense."},
    {"subject": "English Literature", "filename": "doc_12_literature.pdf", "fact": "William Shakespeare was a renowned English playwright, poet, and actor, widely regarded as the greatest writer in the English language.", "question": "Who is widely regarded as the greatest writer in the English language?", "answer": "William Shakespeare is widely regarded as the greatest writer in the English language."},
    {"subject": "Global Geography", "filename": "doc_13_geography.pdf", "fact": "The Amazon River in South America is the largest river in the world by discharge volume of water.", "question": "Which river has the largest discharge volume of water in the world?", "answer": "The Amazon River in South America is the largest river by discharge volume of water."},
    {"subject": "Human Genetics", "filename": "doc_14_genetics.pdf", "fact": "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix.", "question": "What is the structural shape of a DNA molecule?", "answer": "DNA is composed of two polynucleotide chains that coil around each other to form a double helix."},
    {"subject": "Thermodynamics", "filename": "doc_15_chemistry.pdf", "fact": "The freezing point of fresh water is 0 degrees Celsius (or 32 degrees Fahrenheit) at standard atmospheric pressure.", "question": "What is the freezing point of fresh water in Celsius at standard pressure?", "answer": "The freezing point of fresh water is 0 degrees Celsius."},
    {"subject": "Earth's Topography", "filename": "doc_16_mountains.pdf", "fact": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.", "question": "Where is Mount Everest located?", "answer": "Mount Everest is located in the Mahalangur Himal sub-range of the Himalayas."},
    {"subject": "Musical History", "filename": "doc_17_music.pdf", "fact": "The piano is an acoustic, stringed musical instrument invented in Italy by Bartolomeo Cristofori around the year 1700.", "question": "Who invented the piano and approximately when?", "answer": "The piano was invented by Bartolomeo Cristofori around the year 1700."},
    {"subject": "Computer Hardware", "filename": "doc_18_hardware.pdf", "fact": "A Central Processing Unit (CPU) is the primary hardware component of a computer that acts as its brain.", "question": "What is the primary function of a CPU in a computer?", "answer": "A CPU is the primary hardware component that acts as the computer's 'brain.'"},
    {"subject": "Renaissance Art", "filename": "doc_19_art.pdf", "fact": "The Mona Lisa is a world-famous half-length portrait painting created by the Italian polymath Leonardo da Vinci.", "question": "Who painted the Mona Lisa?", "answer": "The Mona Lisa was painted by the Italian polymath Leonardo da Vinci."},
    {"subject": "Botany", "filename": "doc_20_botany.pdf", "fact": "Photosynthesis is the biological process used by plants and other organisms to convert light energy into chemical energy.", "question": "How do plants convert light energy into chemical energy?", "answer": "Plants convert light energy into chemical energy through a biological process called photosynthesis."}
]

def generate_text_block(subject, is_intro=True):
    """Generates standard contextual filler text to bulk up the PDF pages."""
    text = f"The exploration of {subject} represents a significant milestone in our understanding of the domain. Researchers and practitioners have dedicated countless hours to unraveling the complexities inherent in this field. By examining the fundamental principles, we can begin to appreciate the intricate mechanisms that govern its behavior. The historical context surrounding this topic provides a foundational backdrop that is essential for any serious student or professional. Furthermore, the modern applications continue to evolve, pushing the boundaries of what was previously thought possible.\n\n"
    text += f"When analyzing the core components of {subject}, it becomes evident that a multidisciplinary approach yields the most comprehensive insights. Theoretical frameworks often intersect with practical implementations, creating a dynamic ecosystem of innovation. Experts frequently debate the optimal methodologies for further study, yet there is a general consensus regarding its enduring impact on society. As we project into the future, the ongoing discourse will undoubtedly shape the trajectory of emerging technologies and paradigms associated with this subject.\n\n"
    return text

def create_pdf(item):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"A Comprehensive Guide to {item['subject']}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    
    for _ in range(4):
        pdf.multi_cell(0, 8, txt=generate_text_block(item['subject']))
    
    pdf.set_font("Arial", 'B', 12)
    pdf.multi_cell(0, 8, txt=f"\nKEY FACT: {item['fact']}\n")
    pdf.set_font("Arial", size=12)
    
    for _ in range(4):
        pdf.multi_cell(0, 8, txt=generate_text_block(item['subject']))
        
    pdf.output(item['filename'])
    print(f"Generated: {item['filename']} ({pdf.page_no()} pages)")

def main():
    eval_dataset = []
    
    print("Generating 20 multi-page PDF documents...")
    for item in kb_data:
        create_pdf(item)
        
        eval_dataset.append({
            "question": item["question"],
            "answer": item["answer"],
            "source": item["filename"]
        })
        
    with open("../evaluation_dataset.json", "w") as f:
        json.dump(eval_dataset, f, indent=4)
        
    print("\nSuccess! 20 PDFs and 'evaluation_dataset.json' have been created.")

if __name__ == "__main__":
    main()
