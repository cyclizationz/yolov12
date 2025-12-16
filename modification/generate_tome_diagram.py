
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_tokens(ax, x, y, n, color='blue', label="Tokens"):
    for i in range(n):
        rect = patches.Rectangle((x + i*0.6, y), 0.5, 0.5, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
    ax.text(x + n*0.6/2, y - 0.3, label, ha='center', fontsize=9)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# 1. Initial Tokens (Patch Embed)
draw_tokens(ax, 1, 3, 6, color='#aec7e8', label="N Tokens\n(After Patch Embed)")

# Arrow
ax.annotate("", xy=(5, 3.25), xytext=(4.5, 3.25), arrowprops=dict(arrowstyle="->"))

# 2. ToMe Operation
circle = patches.Circle((6, 3.25), 0.8, facecolor='#ffcccc', edgecolor='red')
ax.add_patch(circle)
ax.text(6, 3.25, "ToMe\nMerge\nr=0.25", ha='center', va='center', color='red', fontweight='bold')

# Arrow
ax.annotate("", xy=(7.5, 3.25), xytext=(7, 3.25), arrowprops=dict(arrowstyle="->"))

# 3. Reduced Tokens
draw_tokens(ax, 8, 3, 4, color='#aec7e8', label="0.75N Tokens\n(To Backbone)")

# Explanation text
ax.text(6, 1, "Early merging reduces computational cost\nfor all subsequent Transformer layers.", 
        ha='center', style='italic', fontsize=11, bbox=dict(facecolor='white', alpha=0.5))

plt.title("Token Merging (ToMe) Strategy", fontsize=14)
plt.tight_layout()
plt.savefig('modification/outputs/tome_strategy_diagram.png', dpi=300)
print("Saved modification/outputs/tome_strategy_diagram.png")

