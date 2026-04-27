from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.classical import expr
from qiskit_ibm_runtime.circuit import MidCircuitMeasure


def unitary_evolve(n, h, time, steps) :
    data = QuantumRegister(n, 'data')
    meas = ClassicalRegister(n, 'meas')
    circuit = QuantumCircuit(data, meas)

    trotter_step = QuantumCircuit(n)

    dt = time / steps

    for i in range(0, n-1, 2) :
        trotter_step.rxx(2*dt, i, i+1)
    for i in range(1, n-1, 2) :
        trotter_step.rxx(2*dt, i, i+1)

    for i in range(0, n-1, 2) :
        trotter_step.ryy(2*dt, i, i+1)
    for i in range(1, n-1, 2) :
        trotter_step.ryy(2*dt, i, i+1)

    for i in range(0, n-1, 2) :
        trotter_step.rzz(2*dt, i, i+1)
    for i in range(1, n-1, 2) :
        trotter_step.rzz(2*dt, i, i+1)

    for i in range(n) :
        trotter_step.rz(2*h*dt, i)

    for step in range(steps) :
        circuit.compose(trotter_step, range(n), inplace=True)

    return circuit



def dynamic_trotter_step(n, h, dt, mcm=True, stretch_dd=False, stretch_id='') :
    data = QuantumRegister(n, 'data')
    anc  = QuantumRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')

    trotter_step = QuantumCircuit(data, anc, meas, syn)

    def rzz_step(theta, gate) :
        for i in range(0, n-1, 2) :
            trotter_step.reset(anc[i])
            trotter_step.cx(i, anc[i])
            trotter_step.cx(i+1, anc[i])
            trotter_step.rz(theta, anc[i])
            trotter_step.h(anc[i])

        for i in range(1, n-1, 2) :
            trotter_step.reset(anc[i])
            trotter_step.cx(i, anc[i])
            trotter_step.cx(i+1, anc[i])
            trotter_step.rz(theta, anc[i])
            trotter_step.h(anc[i])

        trotter_step.barrier()
        for i in range(n-1) :
            if mcm :
                trotter_step.append(MidCircuitMeasure(), [anc[i]], [syn[i]])
            else :
                trotter_step.measure(anc[i], syn[i])

        if stretch_dd :
            for i in range(n) :
                s = trotter_step.add_stretch(f"s_{stretch_id}_{gate}_{i}")
                trotter_step.delay(s, i)
                trotter_step.x(i)
                trotter_step.delay(s, i)
                trotter_step.delay(s, i)
                trotter_step.x(i)
                trotter_step.delay(s, i)

        for i in range(0, n-1, 2) :
            flag = expr.lift(syn[i])
            with trotter_step.if_test(flag) :
                trotter_step.z(data[i])
                trotter_step.z(data[i+1])

        for i in range(1, n-1, 2) :
            flag = expr.lift(syn[i])
            with trotter_step.if_test(flag) :
                trotter_step.z(data[i])
                trotter_step.z(data[i+1])
        trotter_step.barrier()

    # RXX
    for i in range(n) :
        trotter_step.h(data[i])
    rzz_step(2*dt, 'rxx')
    for i in range(n) :
        trotter_step.h(data[i])

    # RYY
    for i in range(n) :
        trotter_step.s(data[i])
        trotter_step.h(data[i])
        trotter_step.s(data[i])
    rzz_step(2*dt, 'ryy')
    for i in range(n) :
        trotter_step.sdg(data[i])
        trotter_step.h(data[i])
        trotter_step.sdg(data[i])

    # RZZ
    rzz_step(2*dt, 'rzz')

    # RZ
    for i in range(n) :
        trotter_step.rz(2*h*dt, data[i])

    return trotter_step


def inverse_dynamic_trotter_step(n, h, dt, mcm=True, stretch_dd=False, stretch_id='') :
    data = QuantumRegister(n, 'data')
    anc  = QuantumRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')

    trotter_step = QuantumCircuit(data, anc, meas, syn)

    def rzz_step(theta, gate) :
        for i in range(0, n-1, 2) :
            trotter_step.reset(anc[i])
            trotter_step.cx(i, anc[i])
            trotter_step.cx(i+1, anc[i])
            trotter_step.rz(theta, anc[i])
            trotter_step.h(anc[i])

        for i in range(1, n-1, 2) :
            trotter_step.reset(anc[i])
            trotter_step.cx(i, anc[i])
            trotter_step.cx(i+1, anc[i])
            trotter_step.rz(theta, anc[i])
            trotter_step.h(anc[i])

        trotter_step.barrier()
        for i in range(n-1) :
            if mcm :
                trotter_step.append(MidCircuitMeasure(), [anc[i]], [syn[i]])
            else :
                trotter_step.measure(anc[i], syn[i])

        if stretch_dd :
            for i in range(n) :
                s = trotter_step.add_stretch(f"s_{stretch_id}_{gate}_{i}")
                trotter_step.delay(s, i)
                trotter_step.x(i)
                trotter_step.delay(s, i)
                trotter_step.delay(s, i)
                trotter_step.x(i)
                trotter_step.delay(s, i)

        for i in range(0, n-1, 2) :
            flag = expr.lift(syn[i])
            with trotter_step.if_test(flag) :
                trotter_step.z(data[i])
                trotter_step.z(data[i+1])

        for i in range(1, n-1, 2) :
            flag = expr.lift(syn[i])
            with trotter_step.if_test(flag) :
                trotter_step.z(data[i])
                trotter_step.z(data[i+1])
        trotter_step.barrier()

    # RZ
    for i in range(n) :
        trotter_step.rz(-2*h*dt, data[i])

    # RZZ
    rzz_step(-2*dt, 'rzz')

    # RYY
    for i in range(n) :
        trotter_step.sdg(data[i])
        trotter_step.h(data[i])
        trotter_step.sdg(data[i])
    rzz_step(-2*dt, 'ryy')
    for i in range(n) :
        trotter_step.s(data[i])
        trotter_step.h(data[i])
        trotter_step.s(data[i])

    # RXX
    for i in range(n) :
        trotter_step.h(data[i])
    rzz_step(-2*dt, 'rxx')
    for i in range(n) :
        trotter_step.h(data[i])

    return trotter_step



def dynamic_evolve(n, h, time, steps, mcm=True, stretch_dd=False, folds=1) :
    assert folds in [1,3,5], f'Folds must be either 1, 3 or 5'

    data = QuantumRegister(n, 'data')
    anc  = QuantumRegister(n-1, 'anc')
    meas = ClassicalRegister(n, 'meas')
    syn  = ClassicalRegister(n-1, 'syn')

    circuit = QuantumCircuit(data, anc, meas, syn)

    dt = time / steps

    for step in range(steps) :
        circuit.compose(
                dynamic_trotter_step(
                    n, h, dt, mcm=mcm, stretch_dd=stretch_dd, stretch_id=f'{step}'),
                inplace=True)

    for fold in range(folds//2) :
        for step in range(steps) :
            circuit.compose(
                    inverse_dynamic_trotter_step(
                        n, h, dt, mcm=mcm, stretch_dd=stretch_dd, stretch_id=f'{step}'),
                    inplace=True)
            circuit.compose(
                    dynamic_trotter_step(
                        n, h, dt, mcm=mcm, stretch_dd=stretch_dd, stretch_id=f'{step}'),
                    inplace=True)



    return circuit
