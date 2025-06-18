import esprima

js_code = """
function greet(name) {
    console.log("Hello, " + name);
}

document.getElementById("btn").addEventListener("click", function() {
    alert("Clicked!");
});
"""

# Parse to AST
tree = esprima.parseScript(js_code)

# Traverse top-level body nodes
for node in tree.body:
    print(node.type)
    if node.type == 'FunctionDeclaration':
        print(f"Function: {node.id.name}")
    elif node.type == 'ExpressionStatement':
        print(f"Expression: {node.expression.type}")
