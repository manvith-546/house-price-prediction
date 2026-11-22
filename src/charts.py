"""
charts.py
---------
Small SVG charts for the web app, drawn on the server so the site needs no
JavaScript chart library and works offline. Colours come from CSS classes,
so the charts follow the light / dark theme automatically.
"""
import math
from html import escape

from markupsafe import Markup

from predict import format_price


def _nice_ticks(low, high, count=5):
    """Round tick values that cover low..high (e.g. 0, 50, 100, 150)."""
    span = max(high - low, 1e-9)
    raw = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw),
               default=10 * magnitude)
    start = math.floor(low / step) * step
    ticks, value = [], start
    while value <= high + step * 0.5:
        ticks.append(round(value, 6))
        value += step
    return ticks


def _short_price(lakh):
    if lakh >= 100:
        crore = lakh / 100
        return f"₹{crore:.1f} Cr" if crore < 10 else f"₹{crore:.0f} Cr"
    return f"₹{lakh:.0f} L"


def price_curve_svg(curve, house_sqft, house_price, width=560, height=270):
    """Line chart: predicted price (y) against house size (x), with this house marked."""
    left, right, top, bottom = 58, 18, 18, 46
    plot_w, plot_h = width - left - right, height - top - bottom

    xs = [p["sqft"] for p in curve]
    ys = [p["price_lakh"] for p in curve]
    x_min, x_max = min(xs), max(xs)
    x_ticks = [t for t in _nice_ticks(x_min, x_max, 5) if x_min <= t <= x_max]
    y_ticks = _nice_ticks(0, max(ys) * 1.05, 5)
    if y_ticks[-1] < max(ys):
        y_ticks.append(y_ticks[-1] + (y_ticks[1] - y_ticks[0]))
    y_min, y_max = 0, y_ticks[-1]

    def sx(v):
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Predicted price for sizes from {xs[0]:,.0f} to {xs[-1]:,.0f} sq.ft">']

    # grid + y axis labels
    for t in y_ticks:
        y = sy(t)
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">'
                     f'{escape(_short_price(t))}</text>')
    # x axis labels
    for t in x_ticks:
        x = sx(t)
        parts.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{top + plot_h}" '
                     f'y2="{top + plot_h + 5}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 19}" text-anchor="middle">'
                     f'{t:,.0f}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" '
                 f'y2="{top + plot_h}"/>')
    parts.append(f'<text class="axis-title" x="{left + plot_w / 2:.1f}" y="{height - 6}" '
                 f'text-anchor="middle">Total area (sq.ft)</text>')

    # the curve
    path = " ".join(f'{"M" if i == 0 else "L"}{sx(x):.1f},{sy(y):.1f}' for i, (x, y) in enumerate(zip(xs, ys)))
    parts.append(f'<path class="curve" d="{path}"/>')

    # this house: guide lines + marker
    hx, hy = sx(house_sqft), sy(house_price)
    parts.append(f'<line class="guide" x1="{hx:.1f}" x2="{hx:.1f}" y1="{hy:.1f}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="guide" x1="{left}" x2="{hx:.1f}" y1="{hy:.1f}" y2="{hy:.1f}"/>')
    # prices rise with size, so "above-left" of the point is clear of the line
    if hx - left > 90:
        lx, ly, anchor = hx - 12, hy - 12, "end"
    else:
        lx, ly, anchor = hx + 12, hy + 20, "start"
    parts.append(f'<text class="marker-label" x="{lx:.1f}" y="{ly:.1f}" '
                 f'text-anchor="{anchor}">This house</text>')
    parts.append(f'<circle class="marker" cx="{hx:.1f}" cy="{hy:.1f}" r="6.5"/>')

    # hover targets (bigger than the marks)
    for x, y in zip(xs, ys):
        tip = f"{x:,.0f} sq.ft: {format_price(y)}"
        parts.append(f'<circle class="hit" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="11" '
                     f'data-tip="{escape(tip)}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))
