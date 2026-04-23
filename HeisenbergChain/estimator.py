import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli, PauliList

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, IBMBackend

from pathlib import Path

import json

class MyEstimator :

    def __init__(self,
                 job_name,
                 circuit: QuantumCircuit,
                 parameters: list,
                 observable: SparsePauliOp,
                 sampler: Sampler,
                 service: QiskitRuntimeService,
                 backend: IBMBackend, 
                 ):

        self.job_name = job_name
        self.job_id = None
        self.file_path = Path(f'jobs/{job_name}.json')

        self.sampler = sampler
        self.circuit = circuit
        self.parameters = parameters
        self.observable = observable

        self.service = service
        self.backend = backend

        self.commuting_groups = observable.group_commuting(qubit_wise=True)


    def get_qubit_wise_union(self, pauli_list: PauliList):
        # Perform a logical OR across all X and Z arrays in the list
        union_x = pauli_list.x.any(axis=0)
        union_z = pauli_list.z.any(axis=0)
        # Create the representative Pauli string
        return Pauli((union_z, union_x))


    def add_rotations_and_measurements(self, circuit: QuantumCircuit, pauli: Pauli):
        """Adds basis change gates and measures all qubits."""
        for i, p in enumerate(reversed(pauli.to_label())):
            if p == 'X':
                circuit.h(i)
            elif p == 'Y':
                circuit.sdg(i)
                circuit.h(i)
        circuit.measure_all()
        return circuit


    def submit_sampler_jobs(self):
        # 1. Group commuting Pauli operators to reduce circuit executions
        if self.file_path.is_file() :
            print(f"Sampler jobs already submitted. Result in {self.file_path}.")
            return

        pubs = []
        for group in self.commuting_groups :
            union_pauli  = self.get_qubit_wise_union(group.paulis)
            meas_circuit = self.add_rotations_and_measurements(
                    self.circuit.copy(), union_pauli)
            pm = generate_preset_pass_manager(
                    optimization_level=3, backend=self.backend)
            isa_circuit = pm.run(meas_circuit)
            pubs.append((isa_circuit, self.parameters))

        job = self.sampler.run(pubs)
        self.job_id = job.job_id()

        with open(self.file_path, 'w') as f :
            json.dump(self.job_id, f)


    def evaluate_expectation_value(self) :
        if not self.file_path.is_file() :
            print(f"Sampler jobs not submitted. Job ID must be in {self.file_path}.")
            return None

        with open(self.file_path, 'r') as f :
            file_job_id = json.load(f)

        if not self.job_id :
            self.job_id = file_job_id

        job = self.service.job(self.job_id)
        results = job.result()

        total_expectation = 0.0
        for result, pauli_list in zip(results, self.commuting_groups):
            counts = result.data.meas.get_counts()
            exp_val = self._calculate_exp(counts, pauli_list)
            total_expectation += exp_val

        return np.real(total_expectation)


    def _calculate_exp(self, counts: dict, commuting_pauli_list: SparsePauliOp):
        total_shots = sum(counts.values())
        total_expectation = 0.0
        for pauli_string, coeff in zip(
                commuting_pauli_list.paulis, commuting_pauli_list.coeffs) :
            pauli_mask = ['1' if p != 'I' else '0'
                          for p in pauli_string.to_label()]
            expectation = 0.0
            for bit_string, freq in counts.items() :
                parity = 1
                for b, m in zip(bit_string, pauli_mask) :
                    if m == '1' and b == '1' :
                        parity = -parity
                prob = freq / total_shots
                expectation += prob * parity
            total_expectation += coeff * expectation
        return total_expectation
