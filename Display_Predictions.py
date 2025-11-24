import tkinter as tk
from tkinter import ttk, font as tkfont
import pandas as pd
import os

class PredictionViewer:
    def __init__(self, master, csv_file):
        self.master = master
        self.master.title("Arrhythmia Prediction Viewer")
        self.csv_file = csv_file
       
        self.master.configure(bg="#1e1e1e")        
        self.row_font = tkfont.Font(size=12, weight="bold")
        self.header_font = tkfont.Font(size=13, weight="bold")        
        style = ttk.Style()
        style.theme_use("default")

        style.configure("Treeview",
                        background="#252526",
                        foreground="white",
                        fieldbackground="#252526",
                        rowheight=30,
                        font=self.row_font)

        style.map("Treeview",
                  background=[("selected", "#264f78")])
        
        style.configure("Treeview.Heading",
                        font=self.header_font,
                        background="#3c3c3c",
                        foreground="white",
                        padding=6)
        
        table_frame = tk.Frame(master, bg="#1e1e1e")
        table_frame.pack(fill="both", expand=True)
        self.scroll_y = tk.Scrollbar(table_frame)
        self.scroll_y.pack(side="right", fill="y")

        self.scroll_x = tk.Scrollbar(table_frame, orient="horizontal")
        self.scroll_x.pack(side="bottom", fill="x")        
        self.tree = ttk.Treeview(
            table_frame,
            yscrollcommand=self.scroll_y.set,
            xscrollcommand=self.scroll_x.set,
            selectmode="browse"
        )
        self.tree.pack(fill="both", expand=True)

        self.scroll_y.config(command=self.tree.yview)
        self.scroll_x.config(command=self.tree.xview)

        
        self.allowed_columns = [
            "record", "start", "end",
            "true_label", "true_label_name",
            "pred_label", "pred_label_name",
            "hr"
        ]        
        self.class_colors = {
            "Asystole": "#ff4d4d",
            "Bradycardia": "#ff944d",
            "Tachycardia": "#ffcc00",
            "Normal": "#5cd65c"
        }

        self.load_data()

    
    def load_data(self):
        if not os.path.isfile(self.csv_file):
            print("CSV file not found:", self.csv_file)
            return

        df = pd.read_csv(self.csv_file)
        df = df[self.allowed_columns]
        self.build_table(df)
    
    def build_table(self, df):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = self.allowed_columns
        self.tree["show"] = "headings"
        
        for col in self.allowed_columns:
            self.tree.heading(col, text=col.replace("_", " ").title(), anchor="center")
            self.tree.column(col, width=150, anchor="center")
        
        for _, row in df.iterrows():
            tag_color = row["pred_label_name"] if row["pred_label_name"] in self.class_colors else "Default"

            self.tree.insert("",
                             "end",
                             values=list(row),
                             tags=(tag_color,))
        
        for cls, color in self.class_colors.items():
            self.tree.tag_configure(cls, background=color, foreground="black")

        self.tree.tag_configure("Default", background="#252526", foreground="white")


if __name__ == "__main__":
    root = tk.Tk()
    viewer = PredictionViewer(root, "output_pipeline_force4/predictions_val.csv")
    root.geometry("1200x600")
    root.mainloop()