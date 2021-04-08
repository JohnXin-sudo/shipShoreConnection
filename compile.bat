xelatex "main.tex" -synctex=1 -interaction=nonstopmode -file-line-error -pdf
biber "main.tex"
xelatex "main.tex" -synctex=1 -interaction=nonstopmode -file-line-error -pdf
xelatex "main.tex" -synctex=1 -interaction=nonstopmode -file-line-error -pdf