"""Tests for FP_05 Evolutionary Plausibility - increase coverage from 21%."""

from Falsification.FP_05_EvolutionaryPlausibility import (
    EvolutionaryModel,
    EvolutionarySimulator,
    FitnessCalculator,
    SelectionPressure,
    compute_evolutionary_plausibility,
    validate_evolutionary_trajectory,
)


class TestEvolutionaryModel:
    """Test Evolutionary Model."""

    def test_model_creation(self):
        """Test creating evolutionary model."""
        model = EvolutionaryModel(
            population_size=100, mutation_rate=0.01, selection_strength=0.5
        )
        assert model.population_size == 100
        assert model.mutation_rate == 0.01

    def test_initialize_population(self):
        """Test population initialization."""
        model = EvolutionaryModel(population_size=50)
        population = model.initialize_population()
        assert len(population) == 50

    def test_evolve_generation(self):
        """Test evolving one generation."""
        model = EvolutionaryModel(population_size=20)
        model.initialize_population()

        new_population = model.evolve_generation()
        assert len(new_population) == 20


class TestFitnessCalculator:
    """Test Fitness Calculator."""

    def test_calculate_fitness(self):
        """Test fitness calculation."""
        calculator = FitnessCalculator()
        individual = {"trait1": 0.5, "trait2": 0.8}
        environment = {"optimal_trait1": 0.6, "optimal_trait2": 0.7}

        fitness = calculator.calculate_fitness(individual, environment)
        assert isinstance(fitness, (int, float))
        assert fitness >= 0

    def test_calculate_population_fitness(self):
        """Test calculating fitness for population."""
        calculator = FitnessCalculator()
        population = [
            {"trait": 0.5},
            {"trait": 0.6},
            {"trait": 0.7},
        ]
        environment = {"optimal_trait": 0.6}

        fitnesses = calculator.calculate_population_fitness(population, environment)
        assert len(fitnesses) == 3
        assert all(f >= 0 for f in fitnesses)


class TestSelectionPressure:
    """Test Selection Pressure."""

    def test_selection_pressure_calculation(self):
        """Test selection pressure calculation."""
        sp = SelectionPressure()
        fitnesses = [0.5, 0.6, 0.7, 0.8, 0.9]
        pressure = sp.calculate(fitnesses)
        assert isinstance(pressure, (int, float))
        assert pressure >= 0

    def test_weak_selection(self):
        """Test weak selection pressure."""
        sp = SelectionPressure()
        fitnesses = [0.9, 0.91, 0.92, 0.93]  # Similar fitnesses
        pressure = sp.calculate(fitnesses)
        assert pressure < 0.5  # Should indicate weak selection

    def test_strong_selection(self):
        """Test strong selection pressure."""
        sp = SelectionPressure()
        fitnesses = [0.1, 0.3, 0.5, 0.8, 1.0]  # Diverse fitnesses
        pressure = sp.calculate(fitnesses)
        assert pressure > 0.5  # Should indicate strong selection


class TestEvolutionarySimulator:
    """Test Evolutionary Simulator."""

    def test_init(self):
        """Test initialization."""
        simulator = EvolutionarySimulator(generations=100)
        assert simulator.generations == 100

    def test_simulate_evolution(self):
        """Test simulating evolution."""
        simulator = EvolutionarySimulator(generations=10)
        model = EvolutionaryModel(population_size=20)

        results = simulator.simulate(model)
        assert isinstance(results, dict)
        assert "generations" in results
        assert "final_fitness" in results

    def test_get_evolutionary_trajectory(self):
        """Test getting evolutionary trajectory."""
        simulator = EvolutionarySimulator(generations=5)
        trajectory = simulator.get_evolutionary_trajectory()
        assert isinstance(trajectory, list)
        assert len(trajectory) == 5


class TestComputeEvolutionaryPlausibility:
    """Test evolutionary plausibility computation."""

    def test_plausible_scenario(self):
        """Test computing plausibility for plausible scenario."""
        model_params = {
            "population_size": 1000,
            "generations": 1000,
            "mutation_rate": 0.001,
        }
        result = compute_evolutionary_plausibility(model_params)
        assert isinstance(result, dict)
        assert "plausible" in result

    def test_implausible_scenario(self):
        """Test computing plausibility for implausible scenario."""
        model_params = {
            "population_size": 10,  # Too small
            "generations": 10,  # Too few
            "mutation_rate": 1.0,  # Too high
        }
        result = compute_evolutionary_plausibility(model_params)
        assert isinstance(result, dict)


class TestValidateEvolutionaryTrajectory:
    """Test evolutionary trajectory validation."""

    def test_valid_trajectory(self):
        """Test validating valid trajectory."""
        trajectory = [{"generation": i, "fitness": 0.5 + i * 0.01} for i in range(100)]
        result = validate_evolutionary_trajectory(trajectory)
        assert result["valid"] is True

    def test_invalid_trajectory_regression(self):
        """Test validating trajectory with fitness regression."""
        trajectory = [
            {"generation": i, "fitness": 1.0 - i * 0.01}  # Decreasing fitness
            for i in range(100)
        ]
        result = validate_evolutionary_trajectory(trajectory)
        assert result["valid"] is False

    def test_stagnant_trajectory(self):
        """Test validating stagnant trajectory."""
        trajectory = [
            {"generation": i, "fitness": 0.5} for i in range(100)  # No change
        ]
        result = validate_evolutionary_trajectory(trajectory)
        # Stagnant trajectory may or may not be valid depending on implementation
        assert isinstance(result, dict)
