import spacy
import customtkinter as ctk
from spacy.pipeline import EntityRuler
import sys

# إعدادات المظهر العصري (الوضع الليلي والسمة الزرقاء)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ResearchPaperNER:
    def __init__(self):
        print("جاري تحميل نموذج اللغة (spaCy)...")
        self.nlp = spacy.load("en_core_web_sm")
        self.ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        self._add_rules()

    def _add_rules(self):
        patterns = [
            {"label": "MODEL", "pattern": [{"LOWER": "bert"}]},
            {"label": "MODEL", "pattern": [{"LOWER": "yolo"}]},
            {"label": "MODEL", "pattern": [{"LOWER": "cnn"}]},
            {"label": "MODEL", "pattern": [{"LOWER": "transformer"}]},
            {"label": "MODEL", "pattern": [{"LOWER": "lstm"}]},
            {"label": "MODEL", "pattern": [{"LOWER": "resnet"}]},
            
            {"label": "DATASET", "pattern": [{"LOWER": "scierc"}]},
            {"label": "DATASET", "pattern": [{"LOWER": "imagenet"}]},
            {"label": "DATASET", "pattern": [{"LOWER": "coco"}]},
            {"label": "DATASET", "pattern": [{"LOWER": "squad"}]},
            {"label": "DATASET", "pattern": [{"LOWER": "mnist"}]},
            
            {"label": "METRIC", "pattern": [{"LOWER": "f1-score"}]},
            {"label": "METRIC", "pattern": [{"LOWER": "f1"}, {"LOWER": "score"}]},
            {"label": "METRIC", "pattern": [{"LOWER": "accuracy"}]},
            {"label": "METRIC", "pattern": [{"LOWER": "precision"}]},
            {"label": "METRIC", "pattern": [{"LOWER": "recall"}]},
            {"label": "METRIC", "pattern": [{"LOWER": "bleu"}]}
        ]
        self.ruler.add_patterns(patterns)

    def extract_entities(self, text):
        doc = self.nlp(text)
        extracted = []
        for ent in doc.ents:
            if ent.label_ in ["MODEL", "DATASET", "METRIC"]:
                extracted.append((ent.text, ent.label_))
        return extracted

def calculate_f1_score(ground_truths, predictions):
    """ حساب الدقة الرياضية (F1-score) """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_entities, pred_entities in zip(ground_truths, predictions):
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        true_positives += len(true_set.intersection(pred_set))
        false_positives += len(pred_set - true_set)
        false_negatives += len(true_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def show_modern_popup(abstract_text, extracted_entities, current, total):
    """ واجهة تفاعلية واحترافية للتحكم بالبرنامج """
    app = ctk.CTk()
    app.title("AI Research NER System")
    app.geometry("850x600")
    
    # متغير للتحكم بقرار المستخدم (متابعة أو إلغاء)
    user_decision = {"continue": False}

    def on_next():
        user_decision["continue"] = True
        app.destroy()

    def on_cancel():
        user_decision["continue"] = False
        app.destroy()

    # ربط زر الإغلاق (X) في النافذة بدالة الإلغاء
    app.protocol("WM_DELETE_WINDOW", on_cancel)

    # --- 1. قسم الرأس وشريط التقدم ---
    header_frame = ctk.CTkFrame(app, fg_color="transparent")
    header_frame.pack(fill="x", padx=20, pady=(20, 10))
    
    ctk.CTkLabel(header_frame, text=f"Processing Abstract {current} of {total}", 
                 font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold")).pack(side="left")
    
    progress = ctk.CTkProgressBar(header_frame, width=300, height=12, corner_radius=10)
    progress.pack(side="right", pady=10)
    progress.set(current / total)

    # --- 2. الإطار الرئيسي للنص والنتائج ---
    main_frame = ctk.CTkFrame(app, corner_radius=15, fg_color="#2b2b2b")
    main_frame.pack(padx=20, pady=10, fill="both", expand=True)

    # صندوق النص الأصلي
    ctk.CTkLabel(main_frame, text="📄 Original Abstract Text:", text_color="#aeb6bf",
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold")).pack(anchor="w", padx=20, pady=(15, 5))
    
    textbox = ctk.CTkTextbox(main_frame, height=100, font=ctk.CTkFont(family="Segoe UI", size=15), 
                             corner_radius=8, fg_color="#1e1e1e", border_width=1, border_color="#3e3e3e")
    textbox.pack(fill="x", padx=20, pady=5)
    textbox.insert("0.0", abstract_text)
    textbox.configure(state="disabled")

    # صندوق النتائج
    ctk.CTkLabel(main_frame, text="🎯 Extracted Entities (NER):", text_color="#5dade2",
                 font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold")).pack(anchor="w", padx=20, pady=(20, 5))

    entities_frame = ctk.CTkScrollableFrame(main_frame, height=130, corner_radius=8, fg_color="#1e1e1e", border_width=1, border_color="#3e3e3e")
    entities_frame.pack(fill="x", padx=20, pady=5)

    if not extracted_entities:
        ctk.CTkLabel(entities_frame, text="No targeted entities found.", text_color="#e74c3c", font=ctk.CTkFont(size=14)).pack(anchor="w", pady=5, padx=10)
    else:
        for entity_text, label in extracted_entities:
            color = "#2ecc71" if label == "DATASET" else "#9b59b6" if label == "MODEL" else "#f39c12"
            icon = "📊" if label == "DATASET" else "🧠" if label == "MODEL" else "📈"
            
            ent_label = ctk.CTkLabel(entities_frame, text=f"{icon} [{label}]  ➡  {entity_text}", 
                                     font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), text_color=color)
            ent_label.pack(anchor="w", pady=4, padx=10)

    # --- 3. أزرار التحكم السفلية ---
    footer_frame = ctk.CTkFrame(app, fg_color="transparent")
    footer_frame.pack(fill="x", padx=20, pady=20)

    # زر الإلغاء (أحمر)
    cancel_btn = ctk.CTkButton(footer_frame, text="✖ Cancel & Exit", command=on_cancel, width=150, height=40,
                               fg_color="#c0392b", hover_color="#a93226", font=ctk.CTkFont(weight="bold"))
    cancel_btn.pack(side="left")

    # زر المتابعة (أزرق/أخضر)
    btn_text = "Finish Evaluation ✓" if current == total else "Next Abstract ➡"
    btn_color = "#27ae60" if current == total else "#2980b9"
    hover_color = "#229954" if current == total else "#2471a3"

    next_btn = ctk.CTkButton(footer_frame, text=btn_text, command=on_next, width=180, height=40,
                             fg_color=btn_color, hover_color=hover_color, font=ctk.CTkFont(weight="bold"))
    next_btn.pack(side="right")

    app.mainloop()
    return user_decision["continue"]

# ==========================================
# وحدة التنفيذ الرئيسية
# ==========================================
if __name__ == "__main__":
    ner_system = ResearchPaperNER()

    test_documents = [
        "In this paper, we propose a new CNN architecture. We trained our model on the ImageNet dataset and achieved 92% accuracy.",
        "We evaluate the BERT model using the SciERC dataset. The primary metric is the F1-score.",
        "A novel Transformer approach for object detection on COCO. We report higher precision and recall compared to YOLO.",
        "We used LSTM on the SQuAD dataset for question answering, evaluated by exact match and F1 score."
    ]

    ground_truth_entities = [
        [("CNN", "MODEL"), ("ImageNet", "DATASET"), ("accuracy", "METRIC")],
        [("BERT", "MODEL"), ("SciERC", "DATASET"), ("F1-score", "METRIC")],
        [("Transformer", "MODEL"), ("COCO", "DATASET"), ("precision", "METRIC"), ("recall", "METRIC"), ("YOLO", "MODEL")],
        [("LSTM", "MODEL"), ("SQuAD", "DATASET"), ("F1 score", "METRIC")] 
    ]

    predictions = []
    total_docs = len(test_documents)

    print("\n" + "="*50)
    print("بدء النظام وعرض واجهة المستخدم...")
    print("="*50)

    for i, doc_text in enumerate(test_documents):
        extracted = ner_system.extract_entities(doc_text)
        predictions.append(extracted)
        
        # استدعاء الشاشة وتخزين قرار المستخدم
        keep_going = show_modern_popup(doc_text, extracted, i + 1, total_docs)
        
        # إذا ضغط المستخدم على "إلغاء"، نخرج من الحلقة فوراً
        if not keep_going:
            print("\n[تنبيه]: تم إيقاف المعالجة بناءً على طلب المستخدم.")
            break

    # حساب النتيجة النهائية بناءً على ما تمت معالجته فعلياً فقط!
    processed_count = len(predictions)
    if processed_count > 0:
        f1 = calculate_f1_score(ground_truth_entities[:processed_count], predictions)
        
        print("\n" + "="*50)
        print("--- نتائج التقييم النهائي (Evaluation Metrics) ---")
        print(f"Total Documents Processed: {processed_count} out of {total_docs}")
        print(f"Entity-level F1-Score: {f1:.2f} ({f1*100:.1f}%)")
        
        if f1 > 0.70:
            print(">> نجاح! النظام تجاوز شرط الدقة المطلوب (> 0.70).")
        else:
            print(">> فشل! الدقة أقل من المطلوب.")
        print("="*50)
    else:
        print("لم تتم معالجة أي ملخص لتقييمه.")