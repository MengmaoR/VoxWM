# Native source coordinate interface fix

Lua JSON returned integer-valued coordinates as floats, e.g. `[102.0,1974.0,3.0]` at center `[96,1968,0]` (valid offset `[6,6,3]`). The old Python dtype check rejected them after placement. This is a collector error, not a native flow/Fire failure.

The check now requires finite numeric coordinates, exact integer values and each center offset within1..14. It does not round, clip or alter coordinates. Two targeted tests cover real Lua-style return values and rejection of fractional/nonfinite/outside points.

Initial failed/interrupted traces remain at `outputs/vae_fixed24_native_20261117_no_d2/`. That invocation used127.05 wall seconds. No failed attempt is rewritten as a success.
