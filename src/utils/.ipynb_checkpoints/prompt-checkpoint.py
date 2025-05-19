import sys
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ========== prompt builders ===========================================


def choose_prompt(prompt_choices, default_prompt=""):
    """
    Blocking modal dialog → returns the selected / typed prompt string.
    If the user closes the window without pressing “Apply” we fall back to
    `default_prompt`.
    """
    # ========== prompt selector dialog ==================================
    from PyQt5.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QComboBox,
        QLineEdit, QPushButton, QLabel
    )
    from PyQt5.QtCore import Qt

    app = QApplication(sys.argv)

    class Picker(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("MusicGen prompt selector")
            self.selected = None

            lay = QVBoxLayout(self)

            lay.addWidget(QLabel("Select or type a music prompt:"))

            self.combo = QComboBox()
            self.combo.addItems(prompt_choices)
            lay.addWidget(self.combo)

            self.custom = QLineEdit()
            self.custom.setPlaceholderText("Type custom prompt …")
            lay.addWidget(self.custom)

            apply_btn = QPushButton("Apply")
            apply_btn.clicked.connect(self.on_apply)
            lay.addWidget(apply_btn)

        # ----------------------------------------------------------
        def on_apply(self):
            txt = self.custom.text().strip() or self.combo.currentText()
            self.selected = txt if txt else None
            self.accept()                     # close dialog (sets exec_() return)

    dlg = Picker()
    dlg.exec_()                               # ★ modal – blocks here ★
    return dlg.selected or default_prompt


def choose_prompt_cmd(prompt_list, default_prompt=None):
    print("\nChoose a prompt:")
    for i, opt in enumerate(prompt_list):
        print(f"  [{i}] {opt}")
    try:
        sel = int(input(f"Enter number [default {default_prompt or 0}]: ").strip() or -1)
        if sel < 0 or sel >= len(prompt_list): raise ValueError
    except Exception:
        sel = prompt_list.index(default_prompt) if default_prompt in prompt_list else 0
        print(f"Using default: {prompt_list[sel]}")
    return prompt_list[sel]
