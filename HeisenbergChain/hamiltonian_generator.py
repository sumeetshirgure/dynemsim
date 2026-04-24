from qiskit.quantum_info import SparsePauliOp

from hamlib_snippets import get_hdf5_keys, read_qiskit_hdf5

from itertools import product

def get_heisenberg_hamiltonian(num_sites, h):
    """
    Generates the Heisenberg XXX Hamiltonian for a 1D chain.
    H = sum_{i=0}^{n-2} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) - h sum_{i} Z_i
    """
    ham_list = []
    
    # Iterate through adjacent pairs in the chain
    for i in range(num_sites - 1, 0, -1):
        # Add XX, YY, and ZZ interaction terms for each pair
        for pauli in ["XX", "YY", "ZZ"]:
            ham_list.append((pauli, [i-1, i], 1.0))

    for i in range(num_sites-1, -1, -1) :
        ham_list.append(("Z", [i], h))
            
    # num_qubits ensures the operator is defined for the full system
    return SparsePauliOp.from_sparse_list(ham_list, num_qubits=num_sites)


def get_dynamic_heisenberg_hamiltonian(n, h) :
    base_hamiltonian = get_heisenberg_hamiltonian(n, h)
    Id = SparsePauliOp.from_sparse_list([('I'*(n-1), list(range(n-1)), 1)], num_qubits=n-1)
    hamiltonian = Id ^ base_hamiltonian
    return hamiltonian


if __name__ == '__main__' :
    num_sites = [2, 3, 5, 10]
    couplings = [0, 0.1, 0.5, 2, 3, 5]
    for n, h in product(num_sites, couplings) :
        ref_hamiltonian = read_qiskit_hdf5('heis.hdf5', f'graph-1D-grid-nonpbc-qubitnodes_Lx-{n}_h-{h}')
        H = get_heisenberg_hamiltonian(n, h)
        assert ref_hamiltonian.simplify() == H.simplify(), f"{n}, {h}, {ref_hamiltonian} != {H}"
    print("All tests passed")

    print(get_dynamic_heisenberg_hamiltonian(15, 0.1))
