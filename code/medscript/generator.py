"""Random generator of *valid* MedScript programs.

Programs are assembled from clinically flavoured building blocks (vital-sign
alerts, weight-based dosing, BMI, infusion rate, optional lab values, PHI
handling, a scoring loop ...).  Names, thresholds, ranges and the choice of
blocks are randomised, so the corpus has structural as well as lexical
variety.  Several blocks are deliberately *hard negatives*: they look like the
fault patterns of mutations.py but are correct (guarded optionals, anonymised
sinks, boundary thresholds, zero-guarded divisions)."""

import random

ALT = {
    "weight": ["weight", "wt", "body_weight", "patient_weight"],
    "height": ["height", "ht", "body_height"],
    "temp": ["temp", "temperature", "body_temp"],
    "hr": ["hr", "pulse", "heart_rate"],
    "sbp": ["sbp", "systolic", "bp_sys"],
    "rr": ["rr", "resp_rate"],
    "glucose": ["glucose", "gluc", "blood_sugar"],
    "name": ["patient_name", "pname", "full_name"],
    "pid": ["patient_id", "mrn", "record_id"],
    "volume": ["volume", "fluid_volume", "bag_ml"],
    "duration": ["duration", "infusion_time", "run_hours"],
    "weight_lb": ["weight_lb", "wt_lb"],
}

HIGH_WORDS = {"fever", "hyperthermia", "tachycardia", "hypertension", "overdose", "high",
              "tachypnea", "obesity", "shock", "hyperglycemia"}
LOW_WORDS = {"hypothermia", "bradycardia", "hypotension", "underdose", "low",
             "bradypnea", "underweight", "hypoglycemia"}

COMMENTS = [
    "// triage rules for the admission unit",
    "// ward screening routine",
    "// derived indicators and alerts",
    "// bedside monitor post-processing",
    "// nightly batch check",
    "// generated clinical rule set",
    "// medication safety checks",
]


def fmt(x):
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:g}"


class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.nm = {}
        self.inputs = {}        # canonical -> decl line
        self.consts = []
        self.funcs = []         # list of lists of lines
        self.body = []
        self.used_names = set()
        self.n_uid = 0

    # ---- helpers
    def uid(self, stem):
        self.n_uid += 1
        cand = stem if stem not in self.used_names else f"{stem}{self.n_uid}"
        self.used_names.add(cand)
        return cand

    def name(self, canon):
        if canon not in self.nm:
            self.nm[canon] = self.rng.choice(ALT[canon])
            self.used_names.add(self.nm[canon])
        return self.nm[canon]

    def need(self, canon):
        r = self.rng
        n = self.name(canon)
        if canon in self.inputs:
            return n
        if canon == "weight":
            d = f"input {n}: kg range({fmt(r.choice([0.5, 1, 2, 3]))}, {r.choice([150, 200, 250, 300, 400])});"
        elif canon == "height":
            d = f"input {n}: metre range({r.choice([0.3, 0.4, 0.5])}, {r.choice([2.2, 2.3, 2.4])});"
        elif canon == "temp":
            d = f"input {n}: degC range({r.choice([25, 28, 30, 32])}, {r.choice([41, 42, 43, 44])});"
        elif canon == "hr":
            d = f"input {n}: bpm range({r.choice([10, 20, 30])}, {r.choice([220, 250, 280])});"
        elif canon == "sbp":
            d = f"input {n}: mmHg range({r.choice([30, 40, 50])}, {r.choice([240, 260, 300])});"
        elif canon == "rr":
            d = f"input {n}: bpm range({r.choice([4, 6])}, {r.choice([60, 80])});"
        elif canon == "glucose":
            d = f"input {n}: mg/ml? range(0, {r.choice([15, 20, 25])});"
        elif canon in ("name", "pid"):
            d = f"input {n}: string phi;"
        elif canon == "volume":
            d = f"input {n}: ml range({r.choice([10, 50, 100])}, {r.choice([3000, 5000])});"
        elif canon == "duration":
            d = f"input {n}: hour range({r.choice([0.5, 1])}, {r.choice([12, 24])});"
        elif canon == "weight_lb":
            d = f"input {n}: lb range({r.choice([2, 5])}, {r.choice([600, 700])});"
        else:
            raise KeyError(canon)
        self.inputs[canon] = d
        return n

    def alert(self, cond, word, indent=""):
        return [f"{indent}if ({cond}) {{", f'{indent}    print("ALERT {word}");', f"{indent}}}"]

    # ---- blocks
    def b_dose(self):
        r = self.rng
        w = self.weight_kg or self.need("weight")
        cap = self.uid("MAX_DOSE")
        self.consts.append(f"const {cap}: mg = {r.choice([500, 800, 1000, 1500, 2000])} mg;")
        fn = self.uid(r.choice(["dose_for", "calc_dose", "weight_dose"]))
        wp, rp, dv = r.choice([("w", "rate", "d"), ("wt", "mgkg", "amount"), ("mass", "r", "dose")])
        style = r.random()
        f = [f"func {fn}({wp}: kg, {rp}: mg/kg) -> mg {{",
             f"    let {dv}: mg = {rp} * {wp};",
             f"    if ({dv} > {cap}) {{",
             f"        return {cap};",
             "    }",
             f"    return {dv};",
             "}"]
        self.funcs.append(f)
        dose = self.uid(r.choice(["dose", "dose_mg", "rx"]))
        rate = fmt(r.choice([5, 10, 12.5, 15, 20, 25]))
        self.body.append(f"let {dose}: mg = {fn}({w}, {rate} mg/kg);")
        self.body.append(f"print({dose});")
        if style < 0.5:
            self.body += self.alert(f"{dose} > {fmt(r.choice([300, 400, 600]))} mg", "high_dose")

    def b_bmi(self):
        r = self.rng
        w = self.weight_kg or self.need("weight")
        h = self.need("height")
        fn = self.uid(r.choice(["bmi_of", "calc_bmi"]))
        self.funcs.append([f"func {fn}(w: kg, h: metre) -> kg/metre^2 {{",
                           "    return w / (h * h);", "}"])
        v = self.uid("bmi")
        self.body.append(f"let {v} = {fn}({w}, {h});")
        self.body += self.alert(f"{v} > {fmt(r.choice([30, 35, 40]))} kg/metre^2", "obesity")
        if r.random() < 0.5:
            self.body += self.alert(f"{v} < {fmt(r.choice([17, 18.5]))} kg/metre^2", "underweight")

    def b_temp(self):
        r = self.rng
        t = self.need("temp")
        self.body += self.alert(f"{t} > {r.choice([37.5, 38.0, 38.3, 38.5])} degC", "fever")
        if r.random() < 0.6:
            self.body += self.alert(f"{t} < {r.choice([35.0, 35.5, 36.0])} degC", "hypothermia")

    def b_hr(self):
        r = self.rng
        h = self.need("hr")
        self.body += self.alert(f"{h} > {r.choice([100, 110, 120, 130])} bpm", "tachycardia")
        if r.random() < 0.6:
            self.body += self.alert(f"{h} < {r.choice([40, 45, 50, 55])} bpm", "bradycardia")

    def b_bp(self):
        r = self.rng
        s = self.need("sbp")
        self.body += self.alert(f"{s} < {r.choice([80, 90, 100])} mmHg", "hypotension")
        if r.random() < 0.6:
            self.body += self.alert(f"{s} > {r.choice([140, 160, 180])} mmHg", "hypertension")

    def b_shock(self):
        r = self.rng
        h, s = self.need("hr"), self.need("sbp")
        v = self.uid("shock_index")
        self.body.append(f"let {v} = {h} / {s};")
        self.body += self.alert(f"{v} > {r.choice([0.9, 1.0, 1.2])} bpm/mmHg", "shock")

    def b_combo(self):
        r = self.rng
        t, h = self.need("temp"), self.need("hr")
        self.body += self.alert(f"{h} > {r.choice([100, 110])} bpm && {t} > {r.choice([37.8, 38.0])} degC",
                                "sepsis_risk")

    def b_infusion(self):
        r = self.rng
        v, d = self.need("volume"), self.need("duration")
        rt = self.uid("rate")
        self.body.append(f"let {rt}: ml/hour = {v} / {d};")
        self.body += self.alert(f"{rt} > {fmt(r.choice([150, 250, 300]))} ml/hour", "high_rate")

    def b_lab(self):
        r = self.rng
        g = self.need("glucose")
        thr = fmt(r.choice([1.2, 1.8, 2.5]))
        style = r.randrange(3)
        if style == 0:
            self.body.append(f"if (present({g})) {{")
            self.body += self.alert(f"{g} > {thr} mg/ml", "hyperglycemia", "    ")
            self.body.append("}")
        elif style == 1:
            self.body += self.alert(f"present({g}) && {g} > {thr} mg/ml", "hyperglycemia")
        else:
            self.body.append(f"if (!present({g})) {{")
            self.body.append('    print("lab missing");')
            self.body.append("} else {")
            self.body += self.alert(f"{g} < {fmt(r.choice([0.4, 0.5, 0.6]))} mg/ml", "hypoglycemia", "    ")
            self.body.append("}")

    def b_phi(self):
        r = self.rng
        n = self.need(r.choice(["name", "pid"]))
        style = r.randrange(3)
        if style == 0:
            self.body.append(f"export(anonymize({n}));")
        elif style == 1:
            tag = self.uid("masked")
            self.body.append(f"let {tag} = anonymize({n});")
            self.body.append(f"print({tag});")
        else:
            self.body.append(f'print("record processed");')
            self.body.append(f"export(anonymize({n}));")

    def b_score(self):
        r = self.rng
        s, i = self.uid("score"), self.uid("i")
        self.body += [f"let {s} = 0;", f"let {i} = 0;", f"while ({i} < {r.choice([3, 4, 5])}) {{",
                      f"    {s} = {s} + {i} * {r.choice([1, 2, 3])};",
                      f"    {i} = {i} + 1;", "}"]
        self.body += self.alert(f"{s} > {r.choice([3, 5, 8])}", "review")

    def b_convert(self):
        if self.weight_kg:
            return
        wl = self.need("weight_lb")
        self.weight_kg = self.uid("weight_kg")
        self.body.append(f"let {self.weight_kg}: kg = lb_to_kg({wl});")
        self.body.append(f"print({self.weight_kg});")

    def b_severity(self):
        r = self.rng
        t, h = self.need("temp"), self.need("hr")
        fn = self.uid("severity")
        if r.random() < 0.5:
            self.funcs.append([f"func {fn}(t: degC, h: bpm) -> int {{",
                               f"    if (t > {r.choice([39.0, 39.5])} degC) {{", "        return 3;", "    }",
                               f"    else if (h > {r.choice([120, 130])} bpm) {{", "        return 2;", "    }",
                               "    return 0;", "}"])
        else:
            self.funcs.append([f"func {fn}(t: degC, h: bpm) -> int {{",
                               f"    if (t > {r.choice([38.0, 38.5])} degC) {{", "        return 1;", "    } else {",
                               "        return 0;", "    }", "}"])
        v = self.uid("sev")
        self.body.append(f"let {v} = {fn}({t}, {h});")
        self.body += self.alert(f"{v} >= {r.choice([1, 2])}", "escalate")

    def b_consts(self):
        r = self.rng
        a = self.uid("limit")
        self.consts.append(f"const {a}: mg = {r.choice([2, 3])} * {r.choice([200, 250, 400])} mg;")
        v = self.uid("margin")
        self.body.append(f"let {v}: mg = {a} - {r.choice([50, 100])} mg;")
        self.body.append(f"print({v});")

    def b_guarded_div(self):
        """Hard negative: division by a variable whose range includes zero but is guarded."""
        r = self.rng
        d = self.uid("dur")
        self.inputs["_gd"] = f"input {d}: hour range(0, 24);"
        v = self.need("volume")
        q = self.uid("flow")
        self.body.append(f"if ({d} > 0 hour) {{")
        self.body.append(f"    let {q}: ml/hour = {v} / {d};")
        self.body.append(f"    print({q});")
        self.body.append("}")

    def b_boundary(self):
        """Hard negative: threshold exactly at the declared range boundary."""
        h = self.need("hr")
        lo_hi = self.inputs["hr"].split("range(")[1].rstrip(");").split(", ")
        self.body += self.alert(f"{h} > {lo_hi[1]} bpm", "tachycardia")

    BLOCKS = [("b_dose", 3), ("b_bmi", 2), ("b_temp", 3), ("b_hr", 3), ("b_bp", 2), ("b_shock", 2),
              ("b_combo", 1), ("b_infusion", 2), ("b_lab", 3), ("b_phi", 3), ("b_score", 1),
              ("b_convert", 1), ("b_severity", 2), ("b_consts", 1), ("b_guarded_div", 1),
              ("b_boundary", 1)]

    def build(self, n_blocks=None):
        r = self.rng
        self.weight_kg = None
        n_blocks = n_blocks or r.randint(3, 7)
        pool = [b for b, w in self.BLOCKS for _ in range(w)]
        chosen = []
        while len(chosen) < n_blocks:
            b = r.choice(pool)
            if b not in chosen:
                chosen.append(b)
        if "b_convert" in chosen:                     # conversion must precede dosing
            chosen.remove("b_convert")
            chosen.insert(0, "b_convert")
        for b in chosen:
            getattr(self, b)()
        out = []
        if r.random() < 0.7:
            out.append(r.choice(COMMENTS))
        order = ["weight", "weight_lb", "height", "temp", "hr", "sbp", "rr", "glucose", "volume",
                 "duration", "name", "pid", "_gd"]
        for k in order:
            if k in self.inputs:
                out.append(self.inputs[k])
        out += [""] + self.consts if self.consts else []
        for f in self.funcs:
            out.append("")
            out += f
        out.append("")
        self.body_start = len(out)
        out += self.body
        return out


def generate_valid(rng: random.Random, n_blocks=None):
    b = Builder(rng)
    lines = b.build(n_blocks)
    return lines, b.body_start


def to_source(lines):
    return "\n".join(lines) + "\n"
