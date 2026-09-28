"""
Storyteller - Automated Asset Generator
Generates high-resolution sample visual assets for testing and demonstrations.
Course: IGNOU MCA MCSP-232
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


def generate_images(output_dir: Path | str = "./sample_data/images"):
    """Generates 5 stylized 1920x1080 demonstration images matching sample_script.txt."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    W, H = 1920, 1080

    images_data = [
        {
            "filename": "babbage_analytical_engine.png",
            "title": "Charles Babbage Analytical Engine (1837)",
            "subtitle": "Mechanical calculation gears, punch-card storage, and brass cogs",
            "bg_color": (45, 30, 20),
            "accent": (212, 175, 55),  # Metallic brass / gold
            "shape": "cogs"
        },
        {
            "filename": "alan_turing_cryptography.png",
            "title": "Alan Turing & Universal Machines (1940s)",
            "subtitle": "Enigma cipher breaking, paper tape states, and theoretical AI",
            "bg_color": (15, 25, 45),
            "accent": (0, 210, 255),  # Cyan tech glow
            "shape": "grid"
        },
        {
            "filename": "silicon_microprocessor.png",
            "title": "The Silicon Microprocessor Revolution (1970s)",
            "subtitle": "Integrated circuits, VLSI silicon dies, and personal computers",
            "bg_color": (10, 40, 25),
            "accent": (50, 220, 120),  # Emerald circuit green
            "shape": "circuit"
        },
        {
            "filename": "deep_neural_network.png",
            "title": "Deep Learning & Neural Networks (2010s)",
            "subtitle": "Multilayer perceptrons, backpropagation, and visual convolutions",
            "bg_color": (35, 15, 50),
            "accent": (220, 80, 240),  # Neon violet
            "shape": "nodes"
        },
        {
            "filename": "multimodal_ai_robot.png",
            "title": "Multimodal AI & Digital Storytelling (2020s)",
            "subtitle": "Unified text and vision embeddings, zero-shot multimodal synthesis",
            "bg_color": (18, 30, 60),
            "accent": (255, 180, 50),  # Radiant gold / amber
            "shape": "multimodal"
        }
    ]

    for item in images_data:
        img = Image.new("RGB", (W, H), color=item["bg_color"])
        draw = ImageDraw.Draw(img)

        # Draw thematic geometry
        accent = item["accent"]
        shape = item["shape"]

        if shape == "cogs":
            center_x, center_y = W // 2, H // 2 - 40
            for r in [280, 200, 120]:
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=accent,
                    width=4
                )
            for angle_deg in range(0, 360, 20):
                rad = math.radians(angle_deg)
                x1 = center_x + int(240 * math.cos(rad))
                y1 = center_y + int(240 * math.sin(rad))
                x2 = center_x + int(320 * math.cos(rad))
                y2 = center_y + int(320 * math.sin(rad))
                draw.line([(x1, y1), (x2, y2)], fill=accent, width=6)

        elif shape == "grid":
            center_x, center_y = W // 2, H // 2 - 40
            for x in range(300, W - 300, 80):
                draw.line([(x, 150), (x, H - 250)], fill=(30, 60, 100), width=2)
            for y in range(150, H - 250, 60):
                draw.line([(300, y), (W - 300, y)], fill=(30, 60, 100), width=2)
            draw.rectangle(
                [center_x - 350, center_y - 120, center_x + 350, center_y + 120],
                outline=accent,
                width=4
            )

        elif shape == "circuit":
            center_x, center_y = W // 2, H // 2 - 40
            draw.rectangle(
                [center_x - 220, center_y - 220, center_x + 220, center_y + 220],
                outline=accent,
                width=5
            )
            for offset in [-160, -80, 0, 80, 160]:
                draw.line(
                    [(center_x + offset, center_y - 220), (center_x + offset, center_y - 340)],
                    fill=accent,
                    width=3
                )
                draw.line(
                    [(center_x + offset, center_y + 220), (center_x + offset, center_y + 340)],
                    fill=accent,
                    width=3
                )
                draw.line(
                    [(center_x - 220, center_y + offset), (center_x - 340, center_y + offset)],
                    fill=accent,
                    width=3
                )
                draw.line(
                    [(center_x + 220, center_y + offset), (center_x + 340, center_y + offset)],
                    fill=accent,
                    width=3
                )

        elif shape == "nodes":
            layers = [4, 6, 6, 4]
            layer_xs = [W // 2 - 360, W // 2 - 120, W // 2 + 120, W // 2 + 360]
            node_positions = []
            for l_idx, count in enumerate(layers):
                lx = layer_xs[l_idx]
                spacing = 500 // (count + 1)
                l_nodes = []
                for n_idx in range(count):
                    ny = 220 + (n_idx + 1) * spacing
                    l_nodes.append((lx, ny))
                node_positions.append(l_nodes)

            # Draw synapic links
            for i in range(len(node_positions) - 1):
                for n1 in node_positions[i]:
                    for n2 in node_positions[i + 1]:
                        draw.line([n1, n2], fill=(60, 30, 80), width=1)

            # Draw node circles
            for layer in node_positions:
                for x, y in layer:
                    draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=accent, outline=(255, 255, 255), width=2)

        elif shape == "multimodal":
            center_x, center_y = W // 2, H // 2 - 40
            # Concentric radar pulses
            for r in [80, 160, 240, 320]:
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=(50, 70, 120),
                    width=2
                )
            draw.polygon(
                [(center_x, center_y - 140), (center_x + 160, center_y + 100), (center_x - 160, center_y + 100)],
                outline=accent,
                fill=(25, 45, 80)
            )

        # Draw Typography Header & Subtitle
        try:
            title_font = ImageFont.truetype("arial.ttf", 48)
            sub_font = ImageFont.truetype("arial.ttf", 26)
        except IOError:
            title_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        # Banner at top
        draw.rectangle([100, 50, W - 100, 130], fill=(0, 0, 0, 160), outline=accent, width=2)
        draw.text((W // 2, 75), item["title"], font=title_font, fill=(255, 255, 255), anchor="mm")
        draw.text((W // 2, 110), item["subtitle"], font=sub_font, fill=accent, anchor="mm")

        file_dest = out_path / item["filename"]
        img.save(file_dest, "PNG")
        print(f"Generated sample image: {file_dest.name}")


if __name__ == "__main__":
    generate_images()
