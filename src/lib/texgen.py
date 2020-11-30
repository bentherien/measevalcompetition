def startTable(caption="default",label="def", elements="|X|x|", length = "350pt"):
    return "\\begin{{{}}}[!ht] \\centering \n\\caption{{{}}} \n\\label{{{}}}\n\\begin{{{}}}{{{}}}{{{}}}".format("table",caption,label,"tabularx",length,elements)


def endTable():
    return """\end{tabularx}
\end{table}"""