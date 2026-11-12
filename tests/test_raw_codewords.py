"""Received codewords and RS error positions from decode_signal (PHY v1)."""

import pytest
from synth_v1 import v1_codewords, v1_signal

from hubble_satnet_decoder import constants
from hubble_satnet_decoder.decoder import _rs_decode, decode_signal


def _wrong_symbol(sym, delta):
    """A different symbol, ``delta`` steps away from ``sym`` (wrapping)."""
    return (sym + delta) % constants.NUM_FSK_BINS


def _only_packet(sig):
    packets, _, _ = decode_signal(sig)
    assert len(packets) == 1
    return packets[0]


def test_rs_decode_reports_error_positions():
    _, pdu = v1_codewords()
    rx = list(pdu)
    rx[2] = _wrong_symbol(pdu[2], 1)
    rx[17] = _wrong_symbol(pdu[17], 2)
    decoded, n_corr, err_pos = _rs_decode(rx, constants.RS_N_V1, constants.RS_K_V1)
    assert n_corr == 2
    assert err_pos == [2, 17]
    assert list(decoded) == pdu[:len(decoded)]


def test_rs_decode_failure_has_no_positions():
    _, pdu = v1_codewords()
    rx = [_wrong_symbol(s, 1) for s in pdu]
    _, n_corr, err_pos = _rs_decode(rx, constants.RS_N_V1, constants.RS_K_V1)
    assert (n_corr, err_pos) == (-1, [])


def test_clean_packet_carries_received_codewords():
    header, pdu = v1_codewords()
    pkt = _only_packet(v1_signal())
    assert (pkt["seq_num"], pkt["ntw_id"], pkt["auth_tag"]) == (321, 0x1234ABCD, 0xCAFEF00D)
    assert pkt["header_syms"] == header
    assert pkt["pdu_syms_rx"] == pdu
    assert pkt["header_err_pos"] == []
    assert pkt["pdu_err_pos"] == []


def test_corrected_packet_keeps_the_symbols_as_received():
    header, pdu = v1_codewords()
    header_errors = {4: _wrong_symbol(header[4], 9)}
    pdu_errors = {3: _wrong_symbol(pdu[3], 1), 21: _wrong_symbol(pdu[21], 30)}
    pkt = _only_packet(v1_signal(header_errors=header_errors, pdu_errors=pdu_errors))
    assert pkt["seq_num"] == 321
    assert (pkt["header_n_corr"], pkt["pdu_n_corr"]) == (1, 2)
    assert pkt["header_err_pos"] == [4]
    assert pkt["pdu_err_pos"] == [3, 21]
    assert pkt["header_syms"][4] == header_errors[4]
    assert pkt["pdu_syms_rx"][3] == pdu_errors[3]
    assert pkt["pdu_syms_rx"][21] == pdu_errors[21]
    diff = [i for i, (a, b) in enumerate(zip(pkt["pdu_syms_rx"], pdu)) if a != b]
    assert diff == [3, 21]


def test_pdu_failure_attempt_carries_received_pdu():
    _, pdu = v1_codewords()
    # RS corrects at most (n - k) / 2 symbols, so half the codeword wrong is always too many
    errors = {i: _wrong_symbol(pdu[i], 5) for i in range(len(pdu) // 2)}
    packets, _, attempts = decode_signal(v1_signal(pdu_errors=errors))
    assert packets == []
    fails = [a for a in attempts if a["reason"] == "pdu_fail" and a["channel_num"] == 4]
    assert fails
    att = fails[0]
    assert att["header_n_corr"] == 0
    assert len(att["pdu_syms_rx"]) == len(pdu)
    assert [i for i, (a, b) in enumerate(zip(att["pdu_syms_rx"], pdu)) if a != b] == sorted(errors)


@pytest.mark.parametrize("chipset", sorted(constants.SYNTH_RES))
@pytest.mark.parametrize("channel", [0, 2, 9, 15, 16, 18])
@pytest.mark.parametrize("pkt_len_idx", [0, 1, 2, 3])
def test_every_chipset_channel_and_length(chipset, channel, pkt_len_idx):
    header, pdu = v1_codewords(channel=channel, hop_seq_idx=2, pkt_len_idx=pkt_len_idx)
    n = len(pdu)
    # Corrupt the first and last symbol
    errors = {0: _wrong_symbol(pdu[0], 1), n - 1: _wrong_symbol(pdu[n - 1], 2)}
    pkt = _only_packet(v1_signal(channel=channel, hop_seq_idx=2, pkt_len_idx=pkt_len_idx,
                                 chipset=chipset, pdu_errors=errors))
    assert (pkt["channel_num"], pkt["chipset"], pkt["seq_num"]) == (channel, chipset, 321)
    assert pkt["header_syms"] == header
    assert pkt["pdu_err_pos"] == [0, n - 1]
    assert [i for i, (a, b) in enumerate(zip(pkt["pdu_syms_rx"], pdu)) if a != b] == [0, n - 1]


def test_attempts_carry_pdu_error_positions():
    _, pdu = v1_codewords()
    _, _, attempts = decode_signal(v1_signal(pdu_errors={5: _wrong_symbol(pdu[5], 1)}))
    ok = [a for a in attempts if a["reason"] == "ok"]
    assert ok and all(a["pdu_err_pos"] == [5] and a["pdu_n_corr"] == 1 for a in ok)
