/* Small enhancements; every page still works without JavaScript. */
(function () {
  "use strict";
  var root = document.documentElement;

  /* ---------- light / dark theme ---------- */
  var toggle = document.querySelector("[data-theme-toggle]");
  function isDark() {
    if (root.dataset.theme) return root.dataset.theme === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function updateToggleLabel() {
    if (toggle) toggle.setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
  }
  if (toggle) {
    updateToggleLabel();
    toggle.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
      updateToggleLabel();
    });
  }

  /* ---------- number steppers ---------- */
  document.querySelectorAll(".stepper").forEach(function (stepper) {
    var input = stepper.querySelector("input");
    stepper.querySelectorAll("[data-step]").forEach(function (button) {
      button.addEventListener("click", function () {
        var step = Number(button.dataset.step);
        var min = Number(input.min), max = Number(input.max);
        var value = (Number(input.value) || 0) + step;
        input.value = Math.min(max, Math.max(min, value));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
    });
  });

  /* ---------- area-type explanation follows the selected option ---------- */
  var hint = document.querySelector("[data-area-hint]");
  document.querySelectorAll('input[name="area_type"]').forEach(function (radio) {
    radio.addEventListener("change", function () {
      if (hint && radio.checked) hint.textContent = radio.dataset.help || "";
    });
  });

  /* ---------- table search ---------- */
  document.querySelectorAll("[data-filter-table]").forEach(function (box) {
    var table = document.getElementById(box.dataset.filterTable);
    if (!table) return;
    var empty = table.parentElement.querySelector(".empty-filter");
    box.addEventListener("input", function () {
      var q = box.value.trim().toLowerCase();
      var shown = 0;
      table.querySelectorAll("tbody tr").forEach(function (row) {
        var match = !q || row.dataset.name.indexOf(q) !== -1;
        row.hidden = !match;
        if (match) shown++;
      });
      if (empty) empty.hidden = shown !== 0;
    });
  });

  /* ---------- table sorting ---------- */
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var headers = Array.prototype.slice.call(table.querySelectorAll("thead th"));
    headers.forEach(function (th, index) {
      var button = th.querySelector("button[data-sort]");
      if (!button) return;
      button.addEventListener("click", function () {
        var numeric = button.dataset.sort === "number";
        var current = th.getAttribute("aria-sort");
        var dir = current === "descending" ? "ascending" : current === "ascending" ? "descending"
                : (numeric ? "descending" : "ascending");
        headers.forEach(function (h) { h.setAttribute("aria-sort", "none"); });
        th.setAttribute("aria-sort", dir);
        var body = table.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[index].dataset.value, y = b.cells[index].dataset.value;
          var cmp = numeric ? Number(x) - Number(y) : x.localeCompare(y);
          return dir === "ascending" ? cmp : -cmp;
        });
        rows.forEach(function (row) { body.appendChild(row); });
      });
    });
  });

  /* ---------- chart tooltips (anything with data-tip) ---------- */
  var tip = document.querySelector(".chart-tip");
  if (tip) {
    var place = function (event) {
      var x = event.clientX + 14, y = event.clientY + 14;
      var w = tip.offsetWidth, h = tip.offsetHeight;
      if (x + w > window.innerWidth - 8) x = event.clientX - w - 14;
      if (y + h > window.innerHeight - 8) y = event.clientY - h - 14;
      tip.style.left = x + "px";
      tip.style.top = y + "px";
    };
    document.addEventListener("pointerover", function (event) {
      var target = event.target.closest && event.target.closest("[data-tip]");
      if (!target) return;
      tip.textContent = target.getAttribute("data-tip");
      tip.hidden = false;
      place(event);
    });
    document.addEventListener("pointermove", function (event) {
      if (!tip.hidden) place(event);
    });
    document.addEventListener("pointerout", function (event) {
      var target = event.target.closest && event.target.closest("[data-tip]");
      if (target && !target.contains(event.relatedTarget)) tip.hidden = true;
    });
  }
})();
