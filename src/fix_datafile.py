with open("data-test.txt") as f:
    content = f.read()

content = content.replace("; \n", "\n").replace(";\n", "\n")
with open("data-test.txt", "w") as f:
    f.write(content)
