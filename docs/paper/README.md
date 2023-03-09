## JOSS Paper

Can compile pdf using pandoc:
``` Bash
pandoc paper.md -o paper.pdf

# for less margin
pandoc paper.md -o paper.pdf -V geometry:margin=1in 
```

Once this looks good, can trigger github action once pushed. 
