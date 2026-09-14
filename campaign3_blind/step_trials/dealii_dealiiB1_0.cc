#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <sstream>
#include <iomanip>

using namespace dealii;

class SourceFunction : public Function<2>
{
public:
  SourceFunction(double k_val, double x1_val)
    : Function<2>(2), k(k_val), x1(x1_val) {}

  virtual double value(const Point<2> &p, const unsigned int component = 0) const override
  {
    (void)component;
    return k * numbers::PI * numbers::PI * (x1 - p[0]) * std::cos(numbers::PI * p[1]);
  }

private:
  double k, x1;
};

class InterfaceFluxFunction : public Function<2>
{
public:
  InterfaceFluxFunction(const std::vector<Point<2>> &coords,
                        const std::vector<double> &fluxes,
                        double iface_x)
    : Function<2>(1), coords(coords), fluxes(fluxes), iface_x(iface_x) {}

  virtual double value(const Point<2> &p, const unsigned int component = 0) const override
  {
    (void)component;
    if (std::abs(p[0] - iface_x) > 1e-10)
      return 0.0;

    double y = p[1];
    if (y < coords.front()[1])
      return fluxes.front();
    if (y > coords.back()[1])
      return fluxes.back();

    for (size_t i = 0; i < coords.size() - 1; ++i)
    {
      if (y >= coords[i][1] && y <= coords[i+1][1])
      {
        double t = (y - coords[i][1]) / (coords[i+1][1] - coords[i][1]);
        return fluxes[i] + t * (fluxes[i+1] - fluxes[i]);
      }
    }
    return fluxes.back();
  }

private:
  std::vector<Point<2>> coords;
  std::vector<double> fluxes;
  double iface_x;
};

int main()
{
  std::ifstream infile("solver_input.txt");
  if (!infile.is_open())
  {
    std::cerr << "ERROR: Cannot open solver_input.txt" << std::endl;
    return 1;
  }

  double x0, x1, y0, y1, iface_x, k;
  unsigned int nx, ny, degree;
  int side_flag; // 0 = Dirichlet, 1 = Neumann

  infile >> x0 >> x1 >> y0 >> y1 >> iface_x >> k >> nx >> ny >> degree >> side_flag;

  unsigned int n_iface_points;
  infile >> n_iface_points;

  std::vector<Point<2>> iface_coords(n_iface_points);
  std::vector<double> iface_values(n_iface_points);

  for (unsigned int i = 0; i < n_iface_points; ++i)
  {
    infile >> iface_coords[i][0] >> iface_coords[i][1];
    infile >> iface_values[i];
  }
  infile.close();

  Triangulation<2> triangulation;
  GridGenerator::box(triangulation,
                     Point<2>(x0, y0),
                     Point<2>(x1, y1));
  triangulation.refine_global(nx);

  const unsigned int fe_degree = degree;
  FE_Q<2> fe(fe_degree);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs = dof_handler.n_dofs();
  std::cout << "DOF count = " << n_dofs << std::endl;

  std::vector<types::boundary_id> boundary_ids(2, numbers::invalid_boundary_id);
  unsigned int iface_boundary_id = 1;

  GridTools::flatten_faces_on_hyperplane(triangulation,
                                          Point<2>(iface_x, 0.0),
                                          1e-12,
                                          boundary_ids[iface_boundary_id]);

  std::map<types::boundary_id, Function<2> *> dirichlet_boundary_conditions;
  dirichlet_boundary_conditions[0] = new Functions::ZeroFunction<2>();

  std::map<types::boundary_id, Function<2> *> neumann_boundary_conditions;
  if (side_flag == 1)
  {
    neumann_boundary_conditions[iface_boundary_id] =
      new InterfaceFluxFunction(iface_coords, iface_values, iface_x);
  }

  SourceFunction source_function(k, x1);

  QGauss<2> cell_quadrature(fe_degree + 1);
  QGauss<1> face_quadrature(fe_degree + 1);

  FEEvaluation<2, 2, 2> fe_cell(fe, cell_quadrature, update_values | update_gradients | update_JxW_values);
  FEEvaluation<2, 2, 1> fe_face(fe, face_quadrature, update_values | update_normal_vectors | update_JxW_values);

  SparseMatrix<double> system_matrix;
  Vector<double> system_rhs;
  system_matrix.reinit(n_dofs, n_dofs);
  system_rhs.reinit(n_dofs);

  DoFTools::make_sparsity_pattern(system_matrix, dof_handler, true);
  system_matrix.compress(VectorOperation::add);

  Vector<double> volume_rhs(n_dofs);
  volume_rhs.reinit(n_dofs);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_cell.reinit(cell);

    Tensor<1, 2> current_grad_u(fe.n_components);
    fe_cell.get_function_gradients(0, current_grad_u);

    for (unsigned int q_point = 0; q_point < cell_quadrature.size(); ++q_point)
    {
      const double JxW = fe_cell.JxW(q_point);
      const double source_val = source_function.value(fe_cell.quadrature_point(q_point));

      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        {
          const double grad_phi_i = fe_cell.grad(i, q_point)[0];
          const double grad_phi_j = fe_cell.grad(j, q_point)[0];
          system_matrix.add(cell->vertex(i), cell->vertex(j),
                           k * (grad_phi_i * grad_phi_j) * JxW);
        }
        volume_rhs(cell->vertex(i)) -= source_val * fe_cell.shape_value(i, q_point) * JxW;
      }
    }
  }

  system_matrix.add(volume_rhs);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    for (unsigned int face_no = 0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
    {
      if (cell->face(face_no)->at_boundary())
      {
        const auto boundary_id = cell->face(face_no)->boundary_indicator();

        if (neumann_boundary_conditions.count(boundary_id))
        {
          fe_face.reinit(cell, face_no);
          const auto *neumann_function = neumann_boundary_conditions.at(boundary_id);

          for (unsigned int q_point = 0; q_point < face_quadrature.size(); ++q_point)
          {
            const double JxW = fe_face.JxW(q_point);
            const Tensor<1, 2> normal = fe_face.normal_vector(q_point);
            const double neumann_val = neumann_function->value(fe_face.quadrature_point(q_point));

            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              volume_rhs(cell->vertex(i)) += neumann_val * normal[0] * fe_face.shape_value(i, q_point) * JxW;
            }
          }
        }
      }
    }
  }

  AffineConstraints<double> constraints;
  constraints.initialize(n_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  DoFTools::make_dirichlet_constraints(dof_handler, dirichlet_boundary_conditions, constraints);
  constraints.close();

  Vector<double> solution(n_dofs);
  Vector<double> rhs(n_dofs);
  constraints.distribute(volume_rhs, rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner(system_matrix, 1.2);

  system_matrix.add(volume_rhs);
  constraints.distribute(system_matrix);

  solver.solve(system_matrix, solution, rhs, preconditioner);

  Vector<double> residual(n_dofs);
  system_matrix.vmult(residual, solution);
  residual -= volume_rhs;

  std::vector<Point<2>> interface_nodes;
  std::vector<double> interface_values;
  std::vector<double> interface_fluxes;
  std::vector<double> interface_weights;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    for (unsigned int face_no = 0; face_no < GeometryInfo<2>::faces_per_cell; ++face_no)
    {
      if (cell->face(face_no)->at_boundary() &&
          cell->face(face_no)->boundary_indicator() == iface_boundary_id)
      {
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_face; ++v)
        {
          const auto vertex = cell->face(face_no)->vertex(v);
          const auto point = cell->face(face_no)->point(v);

          auto it = std::find_if(interface_nodes.begin(), interface_nodes.end(),
                                [&](const Point<2> &p) { return p == point; });

          if (it == interface_nodes.end())
          {
            interface_nodes.push_back(point);
            interface_values.push_back(0.0);
            interface_fluxes.push_back(0.0);
            interface_weights.push_back(0.0);
          }

          const unsigned int idx = it - interface_nodes;
          const double trib_length = 0.5 * cell->face(face_no)->measure();

          Vector<double> cell_residual(n_dofs);
          cell->get_dof_indices(cell->vertex_indices());

          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            const unsigned int local_dof = cell->vertex(i);
            const double weight = (local_dof == vertex) ? 1.0 : 0.0;
            if (local_dof == vertex)
            {
              interface_fluxes[idx] -= residual(local_dof) / trib_length;
              interface_weights[idx] += trib_length;
            }
          }
        }
      }
    }
  }

  for (unsigned int i = 0; i < interface_nodes.size(); ++i)
  {
    if (interface_weights[i] > 0)
      interface_fluxes[i] /= interface_weights[i];
  }

  std::sort(interface_nodes.begin(), interface_nodes.end(),
            [](const Point<2> &a, const Point<2> &b) { return a[1] < b[1]; });

  std::vector<double> sorted_values(interface_nodes.size());
  std::vector<double> sorted_fluxes(interface_nodes.size());

  for (unsigned int i = 0; i < interface_nodes.size(); ++i)
  {
    const auto &pt = interface_nodes[i];
    auto it = std::find_if(interface_nodes.begin(), interface_nodes.end(),
                          [&](const Point<2> &p) { return p == pt; });
    const unsigned int idx = it - interface_nodes;
    sorted_values[i] = VectorTools::point_value(dof_handler, solution, pt);
    sorted_fluxes[i] = interface_fluxes[idx];
  }

  std::ofstream outfile("solver_output.txt");
  outfile << std::setprecision(16);
  outfile << interface_nodes.size() << std::endl;

  for (unsigned int i = 0; i < interface_nodes.size(); ++i)
  {
    outfile << interface_nodes[i][0] << " " << interface_nodes[i][1] << " "
            << sorted_values[i] << " " << sorted_fluxes[i] << std::endl;
  }
  outfile.close();

  return 0;
}
