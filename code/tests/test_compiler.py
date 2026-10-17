import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from medscript import compile_source                      # noqa: E402
from medscript.equiv import equivalent                     # noqa: E402
from medscript.generator import generate_valid, to_source  # noqa: E402
from medscript.lexer import tokenize                       # noqa: E402

HEAD = """input w: kg range(2, 300);
input t: degC range(30, 43);
input hr: bpm range(20, 250);
input lab: mg/ml? range(0, 20);
input name: string phi;
input vol: ml range(10, 3000);
input dur: hour range(0, 24);
"""


def codes(src):
    return sorted({d.code for d in compile_source(src, codegen=False).diagnostics})


class LexerTests(unittest.TestCase):
    def test_tokens_and_units(self):
        toks, d = tokenize("let x = 38.5 degC;")
        self.assertEqual([t.kind for t in toks], ["let", "ID", "=", "NUM", "UNIT", ";", "EOF"])
        self.assertFalse(d)

    def test_illegal_char(self):
        self.assertIn("LEX001", codes(HEAD + "let a = 1 @ ;"))

    def test_malformed_number(self):
        self.assertIn("LEX002", codes(HEAD + "let a = 3.4.5;"))

    def test_unterminated_string(self):
        self.assertIn("LEX003", codes(HEAD + 'print("abc);'))


class ParserTests(unittest.TestCase):
    def test_missing_semicolon(self):
        self.assertIn("SYN001", codes(HEAD + "let a = 1\nlet b = 2;"))

    def test_missing_paren(self):
        self.assertIn("SYN001", codes(HEAD + "if (t > 38 degC {\n print(1);\n}"))

    def test_missing_brace_eof(self):
        self.assertIn("SYN003", codes(HEAD + "if (t > 38 degC) {\n print(1);\n"))

    def test_recovery_reports_two_errors(self):
        r = compile_source(HEAD + "let a = ;\nlet b = 2 +;\n", codegen=False)
        self.assertGreaterEqual(len([d for d in r.diagnostics if d.code.startswith("SYN")]), 2)

    def test_semantic_skipped_after_syntax_error(self):
        r = compile_source(HEAD + "let a = ;\nprint(zzz);", codegen=False)
        self.assertFalse(r.sem_ran)


class SemanticTests(unittest.TestCase):
    def test_valid_program(self):
        src = HEAD + 'if (t > 38.0 degC) {\n print("ALERT fever");\n}\n'
        self.assertEqual(codes(src), [])

    def test_undeclared(self):
        self.assertIn("SEM001", codes(HEAD + "print(missing);"))

    def test_redeclaration(self):
        self.assertIn("SEM002", codes(HEAD + "let a = 1;\nlet a = 2;\nprint(a);"))

    def test_uninitialised(self):
        self.assertIn("SEM003", codes(HEAD + "let a: kg;\nprint(a);"))

    def test_assign_after_branches(self):
        src = HEAD + "let a: kg;\nif (t > 38 degC) {\n a = 1 kg;\n} else {\n a = 2 kg;\n}\nprint(a);"
        self.assertNotIn("SEM003", codes(src))

    def test_type_mismatch(self):
        self.assertIn("SEM004", codes(HEAD + 'let a: kg = "x";\nprint(a);'))

    def test_unit_mismatch_add(self):
        self.assertIn("SEM005", codes(HEAD + "let a: kg = w + 3 lb;\nprint(a);"))

    def test_unit_algebra(self):
        src = HEAD + "let rate: mg/kg = 5 mg/kg;\nlet d: mg = rate * w;\nprint(d);"
        self.assertEqual(codes(src), [])

    def test_unit_algebra_division(self):
        src = HEAD + "let f: ml/hour = vol / dur2;\nprint(f);"
        self.assertIn("SEM001", codes(src))      # dur2 undeclared

    def test_condition_not_bool(self):
        self.assertIn("SEM008", codes(HEAD + "if (t) {\n print(1);\n}"))

    def test_range_literal(self):
        self.assertIn("DOM001", codes(HEAD + 'if (t > 380 degC) {\n print("x");\n}'))

    def test_range_boundary_is_ok(self):
        self.assertEqual(codes(HEAD + 'if (hr > 250 bpm) {\n print("x");\n}'), [])

    def test_range_outside_declared(self):
        self.assertIn("DOM001", codes(HEAD + 'if (hr > 290 bpm) {\n print("x");\n}'))

    def test_phi_leak_and_anonymize(self):
        self.assertIn("DOM002", codes(HEAD + "print(name);"))
        self.assertEqual(codes(HEAD + "export(anonymize(name));"), [])

    def test_phi_alias_and_concat(self):
        self.assertIn("DOM002", codes(HEAD + "let a = name;\nexport(a);"))
        self.assertIn("DOM002", codes(HEAD + 'print("p: " + name);'))

    def test_missing_value(self):
        self.assertIn("DOM003", codes(HEAD + "let a = lab * 2;\nprint(a);"))

    def test_present_guard(self):
        src = HEAD + 'if (present(lab)) {\n if (lab > 1 mg/ml) {\n print("x");\n }\n}'
        self.assertEqual(codes(src), [])
        src = HEAD + 'if (present(lab) && lab > 1 mg/ml) {\n print("x");\n}'
        self.assertEqual(codes(src), [])

    def test_early_return_guard(self):
        src = HEAD + "func f() -> int {\n if (!present(lab)) {\n  return 0;\n }\n" \
                     "  let a = lab * 2;\n  print(a);\n  return 1;\n}"
        self.assertEqual(codes(src), [])

    def test_division_by_zero(self):
        self.assertIn("DOM004", codes(HEAD + "let a = w / 0;\nprint(a);"))

    def test_possible_zero_divisor_and_guard(self):
        self.assertIn("DOM005", codes(HEAD + "let a = vol / dur;\nprint(a);"))
        src = HEAD + "if (dur > 0 hour) {\n let a = vol / dur;\n print(a);\n}"
        self.assertEqual(codes(src), [])

    def test_missing_return(self):
        src = HEAD + "func f(x: kg) -> kg {\n if (x > 5 kg) {\n  return x;\n }\n}"
        self.assertIn("SEM007", codes(src))

    def test_all_paths_return(self):
        src = HEAD + "func f(x: kg) -> kg {\n if (x > 5 kg) {\n  return x;\n } else {\n  return 5 kg;\n }\n}"
        self.assertEqual(codes(src), [])

    def test_unreachable_warning(self):
        src = HEAD + "func f(x: kg) -> kg {\n return x;\n print(1);\n}"
        r = compile_source(src, codegen=False)
        self.assertTrue(r.ok)
        self.assertIn("W001", [d.code for d in r.warnings])

    def test_arity(self):
        src = HEAD + "func f(x: kg) -> kg {\n return x;\n}\nlet a = f(w, w);\nprint(a);"
        self.assertIn("SEM006", codes(src))

    def test_inverted_rule_is_not_detected(self):
        src = HEAD + 'if (t < 38.0 degC) {\n print("ALERT fever");\n}'
        self.assertEqual(codes(src), [])          # the case the learned model must catch


class BackendTests(unittest.TestCase):
    def test_constant_folding(self):
        r = compile_source("let a = 2 * 3 + 4;\nprint(a);\n")
        text = repr(r.ir_opt["main"])
        self.assertIn("10", text)
        self.assertLess(r.timings["n_ir_opt"], r.timings["n_ir"])

    def test_dead_branch_removed(self):
        r = compile_source("if (1 > 2) {\n print(1);\n}\nprint(2);\n")
        ops = [i[0] for i in r.ir_opt["main"]]
        self.assertEqual(ops.count("call"), 1)

    def test_optimiser_preserves_behaviour(self):
        for s in range(60):
            lines, _ = generate_valid(random.Random(s))
            r = compile_source(to_source(lines))
            self.assertTrue(r.ok, r.diagnostics)
            ok, env, a, b = equivalent(r, trials=10, seed=s)
            self.assertTrue(ok, (env, a, b))

    def test_example_file(self):
        p = os.path.join(os.path.dirname(__file__), "..", "examples", "triage_ok.med")
        with open(p, encoding="utf-8") as fh:
            self.assertTrue(compile_source(fh.read()).ok)


if __name__ == "__main__":
    unittest.main()
