with open("data.txt") as f:
    content = f.read()

content = content.replace("; \n", "\n").replace(";\n", "\n")
with open("data.txt", "w") as f:
    f.write(content)
