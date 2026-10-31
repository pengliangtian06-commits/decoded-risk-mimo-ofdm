"""A fixed 5G-NR LDPC component via py3gpp, not a complete NR-conformance claim."""

from dataclasses import dataclass
import numpy as np
import py3gpp


@dataclass(frozen=True)
class LDPCConfig:
    information_bits: int = 1024
    target_rate: float = 0.5
    qam_order: int = 16
    max_iterations: int = 20
    redundancy_version: int = 0
    decoder_backend: str = "cpp"


class LDPCCodec:
    def __init__(self, config: LDPCConfig = LDPCConfig()):
        if config.information_bits != 1024 or config.target_rate != 0.5:
            raise ValueError("This study currently validates only 1024 information bits and target rate 1/2")
        if config.qam_order not in [4, 16, 64] or config.max_iterations < 1:
            raise ValueError("Unsupported modulation or decoder iteration budget")
        self.config = config
        self.info = py3gpp.nrDLSCHInfo(config.information_bits, config.target_rate)
        if config.decoder_backend not in ["cpp", "py3gpp"]:
            raise ValueError("Decoder backend must be cpp or py3gpp")
        if config.decoder_backend == "cpp":
            import importlib
            from py3gpp.nrDLSCHInfo import getZarray
            table = getZarray()
            index = next(i for i, values in enumerate(table) if self.info["Zc"] in values)
            module = importlib.import_module("py3gpp.nrLDPCEncode")
            self.base_graph = module._load_basegraph(index, self.info["BGN"]).astype(np.int32)
        self.modulation = {4: "QPSK", 16: "16QAM", 64: "64QAM"}[config.qam_order]
        width = int(np.log2(config.qam_order))
        self.transmitted_bits = int(np.ceil(config.information_bits / config.target_rate / width) * width)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=int)
        if bits.shape != (self.config.information_bits,) or np.any((bits != 0) & (bits != 1)):
            raise ValueError("Input must be one binary transport block of the configured length")
        block = py3gpp.nrCRCEncode(bits, self.info["CRC"])
        segmented = py3gpp.nrCodeBlockSegmentLDPC(block, self.info["BGN"])
        encoded = py3gpp.nrLDPCEncode(segmented, self.info["BGN"], algo="sionna")
        transmitted = py3gpp.nrRateMatchLDPC(encoded, self.transmitted_bits,
                                           self.config.redundancy_version, self.modulation, 1)
        return np.asarray(transmitted, dtype=np.uint8).reshape(-1)

    def decode(self, llr: np.ndarray) -> tuple[np.ndarray, bool, int]:
        llr = np.asarray(llr, dtype=float).reshape(-1)
        if len(llr) != self.transmitted_bits or np.any(~np.isfinite(llr)):
            raise ValueError("LLRs must be finite with the configured transmitted length")
        recovered = py3gpp.nrRateRecoverLDPC(llr, self.config.information_bits,
                                            self.config.target_rate, self.config.redundancy_version,
                                            self.modulation, 1)
        if self.config.decoder_backend == "cpp":
            from . import _core
            result = _core.decode_ldpc(recovered, self.base_graph, int(self.info["Zc"]),
                                      self.config.max_iterations, 0.75, True)
            decoded, iterations = result["bits"], result["iterations"][0]
        else:
            decoded, iterations = py3gpp.nrLDPCDecode(recovered, self.info["BGN"],
                                                    self.config.max_iterations,
                                                    Algorithm="Normalized min-sum", ScalingFactor=0.75)
        block, segment_error = py3gpp.nrCodeBlockDesegmentLDPC(
            decoded, self.info["BGN"], self.config.information_bits + self.info["L"])
        payload, crc = py3gpp.nrCRCDecode(block.astype(int), self.info["CRC"])
        return np.asarray(payload, dtype=np.uint8).reshape(-1), bool(segment_error or np.any(crc != 0)), int(iterations)

    def parameters(self) -> dict:
        return {**self.config.__dict__, "transmitted_bits": self.transmitted_bits,
                "actual_payload_rate": self.config.information_bits / self.transmitted_bits,
                "base_graph": int(self.info["BGN"]), "lifting_size": int(self.info["Zc"]),
                "decoder": "normalized min-sum", "scaling_factor": 0.75,
                "implementation": "py3gpp encoder/tables and independent C++ decoder" if self.config.decoder_backend == "cpp" else "py3gpp",
                "scope": "LDPC component, not complete NR conformance"}
