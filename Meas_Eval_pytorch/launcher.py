import os

for x in os.listdir("generatedData"):
    for fold in range(1,9):
        if x == "quantS.tsv":
            print(x)
            print("python3 main.py run -d generatedData/{} -o output/{} -n {} -e {}".format(x, x[:-5]+"out.tex", "Quantities only", 3, fold))
            os.system("python3 main.py run -d generatedData/{} -o output/{} -n {} -e {} -f {}".format(x, x[:-5]+"out.tex", "\"Quantities only\"", 3, fold))
        elif x == "meS.tsv":
            os.system("python3 main.py run -d generatedData/{} -o output/{} -n {} -e {} -f {}".format(x, x[:-5]+"out.tex", "\"MeasuredEntities only\"", 3, fold))
        elif x == "mpS.tsv":
            os.system("python3 main.py run -d generatedData/{} -o output/{} -n {} -e {} -f {}".format(x, x[:-5]+"out.tex", "\"MeasuredProperties only\"", 3, fold))
        elif x == "qualS.tsv":
            os.system("python3 main.py run -d generatedData/{} -o output/{} -n {} -e {} -f {}".format(x, x[:-5]+"out.tex", "\"Qualifiers only\"", 3, fold))
        elif x == "allS.tsv":
            os.system("python3 main.py run -d generatedData/{} -o output/{} -n {} -e {} -f {}".format(x, x[:-5]+"out.tex", "\"all annotations only\"", 10, fold))



