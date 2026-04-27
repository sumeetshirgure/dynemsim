import random

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister

from qiskit.circuit.classical import expr
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime.circuit import MidCircuitMeasure

def static_cnot_ladder(n) :
    data = QuantumRegister(n, 'data')
    anc  = AncillaRegister(n-1, 'anc')
    c0   = ClassicalRegister(n, 'meas')
    c1   = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, c0, c1)

    for i in range(n-1, 0, -1) :
        qc.cx(data[i-1], data[i])

    return qc

def static_inverse_cnot_ladder(n) :
    data = QuantumRegister(n, 'data')
    anc  = AncillaRegister(n-1, 'anc')
    c0   = ClassicalRegister(n, 'meas')
    c1   = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, c0, c1)

    for i in range(n-1) :
        qc.cx(data[i], data[i+1])

    return qc


def dynamic_cnot_ladder(n, mcm=True, stretch_dd=False, stretch_id='') :
    """
    Implements a dynamic circuit for a CNOT ladder.
    https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.7.023120
    """
    data = QuantumRegister(n, 'data')
    anc  = AncillaRegister(n-1, 'anc')
    c0   = ClassicalRegister(n, 'meas')
    c1   = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, c0, c1)

    for i in range(n-1) : qc.cx(data[i], anc[i])
    for i in range(n-1) : qc.cx(anc[i], data[i+1])
    for i in range(n-1) : qc.h(anc[i])

    qc.barrier()

    if mcm :
        for i in range(n-1) :
            qc.append(MidCircuitMeasure(), [anc[i]], [c1[i]])
    else :
        qc.measure(anc, c1)

    if stretch_dd :
        if not stretch_id :
            raise ValueError(
                    "stretch_id shouldn't be empty, and should be unique")
        for i in range(n) :
            s = qc.add_stretch(f"{stretch_id}_{i}")
            qc.delay(s, data[i])
            qc.x(data[i])
            qc.delay(s, data[i])
            qc.delay(s, data[i])
            qc.x(data[i])
            qc.delay(s, data[i])

    parity = expr.lift(c1[n-2])
    for i in range(n-2, -1, -1) :
        with qc.if_test(parity) :
            qc.z(data[i])
        if i > 0 :
            parity = expr.bit_xor(c1[i-1], parity)

    qc.reset(anc)

    qc.barrier()

    qc.reset(anc)

    return qc


def dynamic_inverse_cnot_ladder(n, mcm=True, stretch_dd=False, stretch_id='') :
    """
    Implements a the inverse of the dynamic circuit for a CNOT ladder.
    https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.7.023120
    """
    data = QuantumRegister(n, 'data')
    anc  = QuantumRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, meas, syn)

    for i in range(n) : qc.h(data[i])

    for i in range(n-1, 0, -1) : qc.cx(data[i], anc[i-1])
    for i in range(n-2, -1, -1) : qc.cx(anc[i], data[i])

    for i in range(n-1) : qc.h(anc[i])

    qc.barrier()

    if mcm :
        for i in range(n-1) :
            qc.append(MidCircuitMeasure(), [anc[i]], [syn[i]])
    else :
        qc.measure(anc, syn)

    if stretch_dd :
        if not stretch_id :
            raise ValueError(
                    "stretch_id shouldn't be empty, and should be unique")
        for i in range(n) :
            s = qc.add_stretch(f"{stretch_id}_{i}")
            qc.delay(s, data[i])
            qc.x(data[i])
            qc.delay(s, data[i])
            qc.delay(s, data[i])
            qc.x(data[i])
            qc.delay(s, data[i])

    parity = expr.lift(syn[0])
    for i in range(1, n) :
        with qc.if_test(parity) :
            qc.z(data[i])
        if i < n - 1 :
            parity = expr.bit_xor(syn[i], parity)

    qc.barrier()

    for i in range(n) : qc.h(data[i])

    qc.reset(anc)

    return qc


def dynamic_hea(n, d, theta, fold_idx, mcm=True, stretch_dd=False) :
    data = QuantumRegister(n, 'data')
    anc  = AncillaRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, meas, syn)

    for i in range(n) :
        qc.ry(theta[i], data[i])
        qc.rz(theta[i+n], data[i])

    for layer in range(1, d+1) :
        entangler = dynamic_cnot_ladder(n, mcm, stretch_dd=stretch_dd, stretch_id=f'U_{fold_idx}_{layer}')
        # entangler = static_cnot_ladder(n)
        qc.compose(entangler, range(n+n-1), range(n+n-1), inplace=True)

        for i in range(n) :
            qc.ry(theta[layer*2*n+i], data[i])
            qc.rz(theta[layer*2*n+i+n], data[i])

    return qc


def inverse_dynamic_hea(n, d, theta, fold_idx, mcm=True, stretch_dd=False) :
    data = QuantumRegister(n, 'data')
    anc  = AncillaRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, meas, syn)

    for layer in range(d, 0, -1) :
        for i in range(n) :
            qc.rz(-theta[layer*2*n+i+n], data[i])
            qc.ry(-theta[layer*2*n+i], data[i])

        entangler = dynamic_inverse_cnot_ladder(n, mcm, stretch_dd=stretch_dd, stretch_id=f'I_{fold_idx}_{layer}')
        # entangler = static_inverse_cnot_ladder(n)
        qc.compose(entangler, range(n+n-1), range(n+n-1), inplace=True)

    for i in range(n) :
        qc.rz(-theta[i+n], data[i])
        qc.ry(-theta[i], data[i])

    return qc



def dynamic_esu2(n, d, mcm=True, stretch_dd=False, folds=1) :
    assert folds in [1,3,5], f'Folds must be either 1, 3 or 5'
    theta = ParameterVector('θ', 2*n*(d+1))

    data = QuantumRegister(n, 'data')
    anc  = AncillaRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')
    qc   = QuantumCircuit(data, anc, meas, syn)

    qc.compose(dynamic_hea(n, d, theta, 0, mcm=mcm, stretch_dd=stretch_dd),
               range(n+n-1), range(n+n-1), inplace=True)
    qc.barrier()

    for fold_idx in range(1, 1+folds // 2) :
        qc.compose(inverse_dynamic_hea(n, d, theta, fold_idx, mcm=mcm, stretch_dd=stretch_dd),
                   range(n+n-1), range(n+n-1), inplace=True)
        qc.barrier()
        qc.compose(dynamic_hea(n, d, theta, fold_idx, mcm=mcm, stretch_dd=stretch_dd),
                   range(n+n-1), range(n+n-1), inplace=True)
        qc.barrier()

    return qc
