from moatless.trajectory import Trajectory


def generate_solution_report(trajectory: Trajectory) -> dict:
    report = {}
    for transition in trajectory.transitions:
        if transition.state.name in ["Finished", "Rejected"]:
            if transition.state.output:
                evaluation_result = transition.state.output.get("evaluation_result")

                if evaluation_result:
                    resolved = evaluation_result.get("resolved", False)
                else:
                    resolved = False

                report[transition.state.id] = {
                    "state": transition.state.name,
                    "resolved": resolved,
                    "solution": transition.state.output.get("diff"),
                    "evaluation_result": evaluation_result
                }

    return report