from scipy.optimize import minimize


def optimize_rbc_operation(
    current_energy,
    current_do,
    current_bod,
    current_flow,
    current_ph
):

    def objective(x):

        pump_speed = x[0]
        aeration_rate = x[1]
        chemical_dose = x[2]

        energy_cost = (
            pump_speed * 0.45
            + aeration_rate * 0.35
        )

        chemical_cost = chemical_dose * 0.20

        bod_penalty = max(0, current_bod - 30) * 15

        do_penalty = max(0, 5 - current_do) * 25

        overload_penalty = max(0, current_flow - 120) * 10

        total_cost = (
            energy_cost
            + chemical_cost
            + bod_penalty
            + do_penalty
            + overload_penalty
        )

        return total_cost

    bounds = [
        (1, 10),   # pump speed
        (3, 15),   # aeration rate
        (0.5, 5),  # chemical dose
    ]

    initial_guess = [5, 8, 2]

    result = minimize(
        objective,
        x0=initial_guess,
        bounds=bounds
    )

    return {
        "optimal_pump_speed": round(result.x[0], 2),
        "optimal_aeration_rate": round(result.x[1], 2),
        "optimal_chemical_dose": round(result.x[2], 2),
        "estimated_operating_cost": round(result.fun, 2)
    }