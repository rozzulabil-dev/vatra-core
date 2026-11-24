document.addEventListener("DOMContentLoaded", () => {
  const copyButtons = document.querySelectorAll("[data-copy]");

  copyButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.getAttribute("data-copy");
      const codeEl = targetId ? document.getElementById(targetId) : null;
      if (!codeEl) return;

      const text = codeEl.textContent.trim();
      const defaultLabel = button.textContent;

      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied!";
        button.classList.add("copied");
      } catch {
        button.textContent = "Copy manually";
      }

      window.setTimeout(() => {
        button.textContent = defaultLabel;
        button.classList.remove("copied");
      }, 1800);
    });
  });
});
