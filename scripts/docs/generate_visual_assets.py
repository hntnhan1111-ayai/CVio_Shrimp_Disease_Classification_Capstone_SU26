#!/usr/bin/env python3
"""Generate CVio SVG brand, hero, team, and gallery assets."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def svg_shell(
    title: str, description: str, body: str, view_box: str = "0 0 1200 600"
) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" role="img"
  aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(description)}</desc>
{body.strip()}
</svg>"""


def generate_brand() -> None:
    logo = svg_shell(
        "CVio shrimp aperture logo",
        "A coral shrimp curls around a cyan camera aperture.",
        """
  <rect width="600" height="600" rx="132" fill="#071A2B"/>
  <circle cx="300" cy="300" r="174" fill="none" stroke="#14B8C4" stroke-width="34"/>
  <path d="M165 324c25-125 197-154 270-42 50 77-9 171-101 163-71-6-101-64-75-112 20-37 73-46 105-17"
        fill="none" stroke="#FF6B6B" stroke-width="38" stroke-linecap="round"/>
  <circle cx="203" cy="268" r="12" fill="#F5E9D3"/>
  <path d="M157 301l-64-44m75 23-36-72m269 155 72 37m-86-15 38 66"
        stroke="#FF6B6B" stroke-width="14" stroke-linecap="round"/>
  <circle cx="324" cy="353" r="34" fill="#0F9D8A"/>
""",
        "0 0 600 600",
    )
    wordmark = svg_shell(
        "CVio wordmark",
        "CVio wordmark with the phrase Aquaculture Vision Intelligence.",
        """
  <rect width="1200" height="280" rx="40" fill="#071A2B"/>
  <text x="72" y="160" fill="#F5E9D3" font-size="128" font-family="Arial, sans-serif" font-weight="700">C</text>
  <text x="170" y="160" fill="#14B8C4" font-size="128" font-family="Arial, sans-serif" font-weight="700">Vio</text>
  <rect x="76" y="190" width="622" height="8" rx="4" fill="#FF6B6B"/>
  <text x="76" y="238" fill="#F5E9D3" opacity=".78" font-size="27" font-family="Arial, sans-serif" letter-spacing="6">AQUACULTURE VISION INTELLIGENCE</text>
""",
        "0 0 1200 280",
    )
    write("docs/assets/brand/cvio-logo.svg", logo)
    write("docs/assets/brand/cvio-wordmark.svg", wordmark)


def generate_hero() -> None:
    hero = svg_shell(
        "CVio — Lightweight Deep Learning for Robust Shrimp Disease Analysis",
        "A mobile camera observes a shrimp while classification probabilities and an instance segmentation contour illustrate the two CVio research tracks.",
        """
  <defs>
    <linearGradient id="water" x1="0" y1="0" x2="1" y2="1">
      <stop stop-color="#071A2B"/><stop offset="1" stop-color="#0B4F6C"/>
    </linearGradient>
    <filter id="soft"><feGaussianBlur stdDeviation="18"/></filter>
  </defs>
  <rect width="1600" height="720" rx="38" fill="url(#water)"/>
  <circle cx="1370" cy="120" r="180" fill="#14B8C4" opacity=".10" filter="url(#soft)"/>
  <circle cx="280" cy="620" r="220" fill="#0F9D8A" opacity=".11" filter="url(#soft)"/>
  <g fill="#14B8C4" opacity=".24">
    <circle cx="88" cy="80" r="5"/><circle cx="148" cy="144" r="3"/><circle cx="1510" cy="568" r="6"/>
    <circle cx="1430" cy="622" r="3"/><circle cx="1190" cy="94" r="4"/><circle cx="990" cy="652" r="3"/>
  </g>
  <g transform="translate(90 116)">
    <rect width="440" height="500" rx="52" fill="#06131F" stroke="#14B8C4" stroke-width="6"/>
    <rect x="28" y="62" width="384" height="360" rx="20" fill="#0B3248"/>
    <rect x="168" y="24" width="104" height="12" rx="6" fill="#0B4F6C"/>
    <circle cx="220" cy="461" r="18" fill="none" stroke="#F5E9D3" opacity=".5" stroke-width="4"/>
    <path d="M95 272c35-126 202-146 273-41 44 66-3 148-84 150-64 1-105-44-91-91 12-39 58-58 94-36"
          fill="none" stroke="#FF6B6B" stroke-width="30" stroke-linecap="round"/>
    <path d="M76 281c42-153 240-177 325-49 52 78-5 179-102 181"
          fill="none" stroke="#14B8C4" stroke-width="5" stroke-dasharray="12 10"/>
    <circle cx="132" cy="227" r="9" fill="#F5E9D3"/>
    <path d="M94 259l-55-35m65 17-29-63" stroke="#FF6B6B" stroke-width="8" stroke-linecap="round"/>
  </g>
  <g transform="translate(610 112)" font-family="Arial, sans-serif">
    <text x="0" y="64" fill="#14B8C4" font-size="28" font-weight="700" letter-spacing="5">GRADUATION CAPSTONE RESEARCH</text>
    <text x="0" y="160" fill="#F5E9D3" font-size="88" font-weight="700">CVio</text>
    <text x="0" y="222" fill="#F5E9D3" font-size="38" font-weight="600">Lightweight Deep Learning for Robust</text>
    <text x="0" y="272" fill="#F5E9D3" font-size="38" font-weight="600">Shrimp Disease Analysis</text>
    <text x="0" y="332" fill="#B9DDE4" font-size="23">Classification + instance segmentation + mobile-oriented inference</text>
    <text x="0" y="370" fill="#B9DDE4" font-size="23">under diverse environmental conditions</text>
    <g transform="translate(0 420)">
      <rect width="260" height="104" rx="18" fill="#0B3248" stroke="#14B8C4"/>
      <text x="24" y="36" fill="#B9DDE4" font-size="18">CLASSIFICATION</text>
      <rect x="24" y="55" width="164" height="12" rx="6" fill="#0B4F6C"/><rect x="24" y="55" width="122" height="12" rx="6" fill="#14B8C4"/>
      <rect x="24" y="77" width="164" height="12" rx="6" fill="#0B4F6C"/><rect x="24" y="77" width="66" height="12" rx="6" fill="#FF6B6B"/>
      <text x="202" y="71" fill="#F5E9D3" font-size="15">p(class)</text>
    </g>
    <g transform="translate(286 420)">
      <rect width="260" height="104" rx="18" fill="#0B3248" stroke="#0F9D8A"/>
      <text x="24" y="36" fill="#B9DDE4" font-size="18">INSTANCE MASK</text>
      <path d="M27 77c24-30 61-32 87-8 25 22 54 17 82-4" fill="none" stroke="#14B8C4" stroke-width="5" stroke-dasharray="8 6"/>
      <circle cx="198" cy="64" r="10" fill="#FF6B6B"/>
    </g>
    <g transform="translate(572 420)">
      <rect width="260" height="104" rx="18" fill="#0B3248" stroke="#F6C453"/>
      <text x="24" y="36" fill="#B9DDE4" font-size="18">MOBILE TARGET</text>
      <rect x="28" y="51" width="48" height="40" rx="7" fill="none" stroke="#F6C453" stroke-width="4"/>
      <path d="M95 72h119" stroke="#0F9D8A" stroke-width="7" stroke-linecap="round"/>
      <path d="M201 61l14 11-14 11" fill="none" stroke="#0F9D8A" stroke-width="6"/>
    </g>
  </g>
""",
        "0 0 1600 720",
    )
    write("docs/assets/hero/cvio-hero.svg", hero)


def avatar(name: str, role: str, accent: str, prop: str, symbol: str) -> str:
    return svg_shell(
        f"Illustrated research avatar for {name}",
        f"An abstract, non-photorealistic CVio laboratory avatar representing {role} with {prop}.",
        f"""
  <rect width="400" height="480" rx="44" fill="#071A2B"/>
  <circle cx="200" cy="150" r="86" fill="#F5E9D3"/>
  <path d="M116 144c8-76 160-98 173 10v26H112z" fill="{accent}"/>
  <circle cx="168" cy="157" r="7" fill="#071A2B"/><circle cx="232" cy="157" r="7" fill="#071A2B"/>
  <path d="M174 195q26 20 52 0" fill="none" stroke="#071A2B" stroke-width="6" stroke-linecap="round"/>
  <path d="M95 425v-88c0-61 46-103 105-103s105 42 105 103v88" fill="{accent}"/>
  <rect x="124" y="285" width="152" height="108" rx="12" fill="#0B3248" stroke="#14B8C4" stroke-width="4"/>
  <text x="200" y="353" text-anchor="middle" fill="#F5E9D3" font-size="44" font-family="Arial, sans-serif">{symbol}</text>
  <rect y="414" width="400" height="66" fill="#06131F"/>
  <text x="200" y="441" text-anchor="middle" fill="#F5E9D3" font-size="20" font-family="Arial, sans-serif" font-weight="700">{html.escape(name)}</text>
  <text x="200" y="465" text-anchor="middle" fill="#B9DDE4" font-size="14" font-family="Arial, sans-serif">{html.escape(role)}</text>
""",
        "0 0 400 480",
    )


def generate_team() -> None:
    people = [
        (
            "mentor-vinh",
            "Mentor Vinh",
            "Supervisor / Research Mentor",
            "#F6C453",
            "review board",
            "✓?",
        ),
        ("nhan", "Nhan", "Team Leader", "#14B8C4", "project map", "⑂"),
        ("phong", "Phong", "Research Member", "#0F9D8A", "Fourier spectrum", "∿"),
        ("nhu", "Nhu", "Research Member", "#FF6B6B", "segmentation mask", "▧"),
        ("member-4", "TBD", "Team Member", "#0B4F6C", "dataset panel", "{ }"),
    ]
    for slug, name, role, accent, prop, symbol in people:
        write(f"docs/assets/team/{slug}.svg", avatar(name, role, accent, prop, symbol))
    lineup_items = []
    for index, (_, name, role, accent, _, symbol) in enumerate(people):
        x = 45 + index * 225
        lineup_items.append(
            f'<rect x="{x}" y="70" width="190" height="330" rx="28" fill="#0B3248" stroke="{accent}"/>'
            f'<circle cx="{x + 95}" cy="164" r="58" fill="#F5E9D3"/>'
            f'<path d="M{x + 39} 164q15-75 112-8v30H{x + 36}z" fill="{accent}"/>'
            f'<text x="{x + 95}" y="278" text-anchor="middle" fill="#14B8C4" font-size="40" font-family="Arial, sans-serif">{html.escape(symbol)}</text>'
            f'<text x="{x + 95}" y="340" text-anchor="middle" fill="#F5E9D3" font-size="20" font-weight="700" font-family="Arial, sans-serif">{html.escape(name)}</text>'
            f'<text x="{x + 95}" y="370" text-anchor="middle" fill="#B9DDE4" font-size="13" font-family="Arial, sans-serif">{html.escape(role)}</text>'
        )
    write(
        "docs/assets/team/team-lineup.svg",
        svg_shell(
            "CVio research team lineup",
            "Five abstract laboratory avatars for Mentor Vinh, Nhan, Phong, Nhu, and the fourth student whose name remains to be verified.",
            '<rect width="1200" height="470" rx="36" fill="#071A2B"/>'
            '<text x="600" y="45" text-anchor="middle" fill="#F5E9D3" font-size="26" font-family="Arial, sans-serif">THE CVio RESEARCH TEAM</text>'
            + "".join(lineup_items),
            "0 0 1200 470",
        ),
    )


def generate_gallery() -> None:
    write(
        "docs/assets/diagrams/reviewer-2-shrimp.svg",
        svg_shell(
            "Reviewer number two shrimp",
            "A playful original shrimp reviewer asks where the ablation study is.",
            """
  <rect width="1000" height="420" rx="34" fill="#071A2B"/>
  <path d="M105 233c35-121 197-137 264-35 42 64-7 140-83 137-55-2-89-43-74-83 15-39 62-52 94-25"
        fill="none" stroke="#FF6B6B" stroke-width="34" stroke-linecap="round"/>
  <circle cx="141" cy="191" r="10" fill="#F5E9D3"/>
  <circle cx="363" cy="159" r="58" fill="none" stroke="#14B8C4" stroke-width="12"/>
  <path d="M405 201l75 75" stroke="#14B8C4" stroke-width="18" stroke-linecap="round"/>
  <rect x="518" y="78" width="410" height="238" rx="30" fill="#0B3248" stroke="#0F9D8A" stroke-width="4"/>
  <path d="M518 236l-67 38 67-5" fill="#0B3248" stroke="#0F9D8A" stroke-width="4"/>
  <text x="560" y="145" fill="#14B8C4" font-size="23" font-family="Arial, sans-serif" font-weight="700">REVIEWER #2 SHRIMP</text>
  <text x="560" y="220" fill="#F5E9D3" font-size="34" font-family="Arial, sans-serif">Where is the</text>
  <text x="560" y="265" fill="#FF6B6B" font-size="42" font-family="Arial, sans-serif" font-weight="700">ablation study?</text>
""",
            "0 0 1000 420",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    generate_brand()
    generate_hero()
    generate_team()
    generate_gallery()
    print("Generated CVio SVG assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
