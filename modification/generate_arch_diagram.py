
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color='lightblue', edge='black'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", linewidth=1, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return x+w, y+h/2

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->"))

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Main Backbone Flow
x = 0.5
y = 4
w = 1.5
h = 1

# Input
draw_box(ax, x, y, w, h, "Input Image\n640x640", color='#e0e0e0')
draw_arrow(ax, x+w, y+h/2, x+w+0.5, y+h/2)
x += 2

# Patch Embed
draw_box(ax, x, y, w, h, "Patch Embed\n(Stem)", color='#fff2cc')
draw_arrow(ax, x+w, y+h/2, x+w+0.5, y+h/2)
x += 2

# Stage 1 (PPC highlighted)
draw_box(ax, x, y, w, h, "Stage 1\n(C2PSA/A2C2f)", color='#d1e7dd')
ax.text(x+w/2, y-0.3, "PPC Applied Here", ha='center', color='green', fontsize=9)
draw_arrow(ax, x+w, y+h/2, x+w+0.5, y+h/2)
x += 2

# Stage 2
draw_box(ax, x, y, w, h, "Stage 2\n(C2PSA)", color='#d1e7dd')
draw_arrow(ax, x+w, y+h/2, x+w+0.5, y+h/2)
x += 2

# Stage 3
draw_box(ax, x, y, w, h, "Stage 3\n(C2PSA)", color='#d1e7dd')
draw_arrow(ax, x+w, y+h/2, x+w+0.5, y+h/2)
x += 2

# Head
draw_box(ax, x, y, w, h, "Seg Head", color='#f4cccc')

# ToMe Annotation
# Arrow pointing after Patch Embed
ax.annotate("ToMe Insertion Point\n(Optimal)", xy=(3.5, 5.1), xytext=(3.5, 6.5),
            arrowprops=dict(facecolor='red', shrink=0.05),
            ha='center', color='red', fontweight='bold')

# PPC Detail Bubble
# Draw a zoom-in for Stage 1
detail_x = 2
detail_y = 1
detail_w = 4
detail_h = 2
rect = patches.Rectangle((detail_x, detail_y), detail_w, detail_h, linewidth=1, edgecolor='green', facecolor='none', linestyle='--')
ax.add_patch(rect)
ax.text(detail_x+0.2, detail_y+detail_h-0.3, "Area Attention Block", fontsize=9, color='green', fontweight='bold')

# Inside detail: AAttn structure
sub_x = detail_x + 0.5
sub_y = detail_y + 0.5
sub_w = 1
sub_h = 0.8
draw_box(ax, sub_x, sub_y, sub_w, sub_h, "Q K V", color='white')
draw_arrow(ax, sub_x+sub_w, sub_y+sub_h/2, sub_x+sub_w+0.5, sub_y+sub_h/2)

draw_box(ax, sub_x+1.5, sub_y, 1.5, sub_h, "PPC (5x5)\nConv", color='#90ee90')

# Connect detail to Stage 1
ax.plot([2.75, 2.75], [4, 3], color='green', linestyle='--')

plt.title("YOLOv12 Architecture & Modifications", fontsize=14)
plt.tight_layout()
plt.savefig('modification/outputs/yolov12_arch_diagram.png', dpi=300)
print("Saved modification/outputs/yolov12_arch_diagram.png")

