import json

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

eval_dataset = []
for item in kb_data:
    eval_dataset.append({
        "question": item["question"],
        "answer": item["answer"],
        "source": item["filename"]
    })

with open("data/evaluation_dataset.json", "w") as f:
    json.dump(eval_dataset, f, indent=4)

print("Restored evaluation_dataset.json")
