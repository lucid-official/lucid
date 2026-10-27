"""Unit tests for ``tools/engine_symdiff.py``."""

from pathlib import Path

from tools import engine_symdiff

_SAMPLE_OBJDUMP_A = """\
000000000007e54c <__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE.llvm.123>:
   7e54c:      	sub	sp, sp, #0xb0
   7e550:      	b.ne	0x7e5ec <__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE.llvm.123+0xa0>
   7e554:      	bl	0x7a628 <__ZN5lucid7backend10CpuBackend5zerosERKNS_5ShapeE>
   7e558:      	adrp	x9, 0x84a000 <__ZTVN5lucid7backend10CpuBackendE+0x558>
   7e55c:      	add	x9, x9, #0x2c0
   7e560:      	adrp	x10, 0x910000 <__ZL12kTable>
   7e564:      	ldr	x8, [x10, #0x48]
   7e568:      	bl	0x80100 <__ZZN5lucid7backend10CpuBackend3addEENK3$_7clEf>
   7e56c:      	nop
   7e570:      	ret

0000000000900000 <__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE.cold.1>:
  900000:      	brk	#0x1
"""

_SAMPLE_OBJDUMP_B = """\
000000000018a000 <__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE.llvm.999>:
  18a000:      	sub	sp, sp, #0xb0
  18a004:      	b.ne	0x18a0a0 <__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE.llvm.999+0xa0>
  18a008:      	bl	0x110000 <__ZN5lucid7backend10CpuBackend5zerosERKNS_5ShapeE>
  18a00c:      	adrp	x9, 0x999000 <__ZTVN5lucid7backend10CpuBackendE+0x558>
  18a010:      	add	x9, x9, #0x7f0
  18a014:      	adrp	x10, 0xa20000 <__ZL12kTable>
  18a018:      	ldr	x8, [x10, #0x90]
  18a01c:      	bl	0x220400 <__ZZN5lucid7backend10CpuBackend3addEENK3$_42clEf>
  18a020:      	ret
"""


def test_normalize_symbol_name_strips_cold_llvm_and_lambda_numbers() -> None:
    raw = "__ZZN5lucid7backend10CpuBackend3addEENK4$_42clEf.llvm.881.cold.2"
    assert (
        engine_symdiff.normalize_symbol_name(raw) == "__ZZN5lucid7backend10CpuBackend3addEENK$_clEf"
    )


def test_parse_objdump_output_normalizes_addresses_branches_adrp_and_nops() -> None:
    parsed_a = engine_symdiff.parse_objdump_output(_SAMPLE_OBJDUMP_A)
    parsed_b = engine_symdiff.parse_objdump_output(_SAMPLE_OBJDUMP_B)

    key = "__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE"
    assert list(parsed_a.keys()) == [key]
    assert parsed_a[key] == parsed_b[key]
    assert parsed_a[key] == [
        "sub\tsp, sp, #0xb0",
        "b.ne\t<SELF+0xa0>",
        "bl\t<__ZN5lucid7backend10CpuBackend5zerosERKNS_5ShapeE>",
        "adrp\tx9, PAGE",
        "add\tx9, x9, PAGEOFF",
        "adrp\tx10, PAGE",
        "ldr\tx8, [x10, PAGEOFF]",
        "bl\t<__ZZN5lucid7backend10CpuBackend3addEENK$_clEf>",
        "ret",
    ]


def test_parse_objdump_detects_real_instruction_change() -> None:
    parsed_a = engine_symdiff.parse_objdump_output(_SAMPLE_OBJDUMP_A)
    mutated = _SAMPLE_OBJDUMP_B.replace("sub\tsp, sp, #0xb0", "sub\tsp, sp, #0xc0")
    parsed_m = engine_symdiff.parse_objdump_output(mutated)
    key = "__ZN5lucid7backend10CpuBackend3negERKNS_7StorageE"
    assert parsed_a[key] != parsed_m[key]


def test_self_comparison_on_engine_so() -> None:
    root = Path(__file__).resolve().parents[3]
    so_candidates = sorted((root / "lucid" / "_C").glob("engine.cpython-*.so"))
    assert so_candidates, "engine .so not found in lucid/_C"
    so = so_candidates[0]
    rc = engine_symdiff.main([str(so), str(so)])
    assert rc == 0
