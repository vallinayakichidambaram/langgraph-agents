import tinycss2

rules = tinycss2.parse_stylesheet("#cell div { width: 50% }")
print("prelude")
print(rules[0].prelude)
print("content")
print(rules[0].content)

print(rules[0].serialise())