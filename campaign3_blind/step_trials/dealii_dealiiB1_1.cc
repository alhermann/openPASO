// solver.cc - deal.II solver for subdomain B (Neumann side)
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>

using namespace dealii;

class SourceTerm : public Function<2>
{
public:
  SourceTerm(double k_coef, double x_outer)
    : Function<2>(2), k(k_coef), x_outer_val(x_outer) {}
  
  virtual double value(const Point<2> &p, unsigned int component = 0) const override
  {
    // f(x, y) = 5*pi^2*(1.4 - x)*cos(pi*y)
    (void)component;
    return k * std::pow(std::acos(-1.0), 2) * (x_outer_val - p[0]) * std::cos(std::acos(-1.0) * p[1]);
  }
  
private:
  double k;
  double x_outer_val;
};

class InterfaceFunction : public Function<2>
{
public:
  InterfaceFunction(std::vector<double> ys, std::vector<double> vals)
    : Function<2>(2), y_coords(ys), values(vals) {}
  
  virtual double value(const Point<2> &p, unsigned int component = 0) const override
  {
    (void)component;
    double y = p[1];
    if (y <= y_coords.front()) return values.front();
    if (y >= y_coords.back()) return values.back();
    
    for (size_t i = 0; i < y_coords.size() - 1; ++i)
    {
      if (y >= y_coords[i] && y <= y_coords[i+1])
      {
        double t = (y - y_coords[i]) / (y_coords[i+1] - y_coords[i]);
        return values[i] + t * (values[i+1] - values[i]);
      }
    }
    return values.back();
  }
  
private:
  std::vector<double> y_coords;
  std::vector<double> values;
};

int main()
{
  try
  {
    // Read input file
    std::ifstream infile("input.txt");
    if (!infile.is_open())
    {
      std::cerr << "ERROR: Cannot open input.txt" << std::endl;
      return 1;
    }
    
    double x0, x1, y0, y1, k_val, iface_x;
    unsigned int nx, ny, degree;
    std::vector<double> iface_ys, iface_vals;
    
    infile >> x0 >> x1 >> y0 >> y1 >> k_val >> iface_x >> nx >> ny >> degree;
    
    unsigned int n_iface = 0;
    infile >> n_iface;
    for (unsigned int i = 0; i < n_iface; ++i)
    {
      double y, v;
      infile >> y >> v;
      iface_ys.push_back(y);
      iface_vals.push_back(v);
    }
    infile.close();
    
    // Setup triangulation
    Triangulation<2> triangulation;
    GridGenerator::hyper_rectangle(triangulation,
                                    Point<2>(x0, y0),
                                    Point<2>(x1, y1));
    triangulation.refine_global(nx);
    
    // Setup DoFHandler
    FE_Q<2> fe(degree);
    DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    
    unsigned int n_dofs = dof_handler.n_dofs();
    std::cout << "DOF count = " << n_dofs << std::endl;
    
    // Setup sparsity pattern
    std::vector<DoFHandler<2>::active_cell_iterator> cells;
    for (auto &cell : dof_handler.active_cell_iterators())
      cells.push_back(cell);
    
    SparsityPattern sparsity_pattern;
    DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, true);
    sparsity_pattern.compress();
    
    // Setup quadrature
    const unsigned int n_q_points = (degree + 1) * (degree + 1);
    QGauss<2> quadrature(n_q_points);
    QGauss<1> face_quadrature(degree + 1);
    
    // Setup FEFaceValues for boundary integrals
    FEFaceValues<2> face_values(fe, face_quadrature);
    
    // Create source term
    SourceTerm source_term(k_val, x1);
    
    // Create interface function for Neumann BC
    InterfaceFunction iface_func(iface_ys, iface_vals);
    
    // Assemble system WITHOUT constraints (for consistent flux recovery)
    SparsityPattern free_sparsity_pattern;
    DoFTools::make_sparsity_pattern(dof_handler, free_sparsity_pattern, true);
    free_sparsity_pattern.compress();
    
    SparseMatrix<double> free_matrix;
    free_matrix.initialize(free_sparsity_pattern);
    Vector<double> free_rhs(n_dofs);
    
    Vector<double> cell_rhs(n_dofs);
    
    FEFaceValues<2> face_values_face(fe, face_quadrature);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_rhs = 0;
      FEValues<2> fe_values(cell->get_fe(), quadrature,
                           update_values | update_grads | update_JxW_values | update_quadrature_points);
      
      fe_values.reinit(cell);
      
      const unsigned int n_dofs_per_cell = fe_values.n_dofs_per_cell();
      const unsigned int n_q_points = fe_values.n_quadrature_points();
      
      const std::vector<double> &JxW_values = fe_values.JxW_values();
      const std::vector<Tensor<1,2>> &grad_phi = fe_values.grad;
      const std::vector<Point<2>> &q_points = fe_values.quadrature_points;
      
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        double f_val = source_term.value(q_points[q]);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        {
          cell_rhs(i) += f_val * fe_values.shape_value(i, q) * JxW_values[q];
        }
      }
      
      // Stiffness matrix: k * grad u * grad v
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        {
          double stiffness = 0;
          for (unsigned int q = 0; q < n_q_points; ++q)
            stiffness += k_val * (grad_phi[i][q] * grad_phi[j][q]) * JxW_values[q];
          free_matrix.add(i, j, stiffness);
        }
      
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        free_rhs(cell->vertex(i)) += cell_rhs(i);
    }
    
    // Apply Neumann BC on interface (x = iface_x)
    // This is the partner's flux - we apply it as natural BC
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int face_no : cell->face_indices())
      {
        const auto &face = cell->face(face_no);
        
        // Check if this face is on the interface (x = iface_x)
        if (face->at_boundary())
        {
          Point<2> center = face->center();
          if (std::abs(center[0] - iface_x) < 1e-12)
          {
            face_values_face.reinit(cell, face_no);
            
            const unsigned int n_face_q_points = face_values_face.n_quadrature_points();
            const std::vector<Tensor<1,2>> &normals = face_values_face.normal_vectors;
            const std::vector<double> &JxW_values = face_values_face.JxW_values;
            const std::vector<double> &shape_values = face_values_face.shape_values;
            const std::vector<Point<2>> &q_points = face_values_face.quadrature_points;
            
            // Neumann: integrate g * v ds where g = partner's flux
            // Partner's flux is outward from partner, so it's inward to us
            // We need to apply it as our natural BC
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              double g_val = iface_func.value(q_points[q]);
              // Flux is in x-direction (normal to interface)
              // Our outward normal on left face (x=iface_x) is (-1, 0)
              // Partner's outward flux is in +x direction, so it's -g_val for our BC
              double g_eff = -g_val; // Sign: partner's outward = our inward
              
              for (unsigned int i = 0; i < face->n_dofs_per_face(); ++i)
              {
                free_rhs(face->vertex(i)) += g_eff * shape_values[i][q] * JxW_values[q];
              }
            }
          }
        }
      }
    }
    
    // Apply Dirichlet BC on x = x1 (outer boundary)
    std::map<types::global_dof_index, double> constraint_values;
    std::vector<bool> is_dirichlet(n_dofs, false);
    
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      for (unsigned int face_no : cell->face_indices())
      {
        const auto &face = cell->face(face_no);
        if (face->at_boundary())
        {
          Point<2> center = face->center();
          if (std::abs(center[0] - x1) < 1e-12)
          {
            for (unsigned int i = 0; i < face->n_dofs_per_face(); ++i)
            {
              types::global_dof_index dofi = face->vertex(i);
              constraint_values[dofi] = 0.0; // u = 0 on outer boundary
              is_dirichlet[dofi] = true;
            }
          }
        }
      }
    }
    
    // Apply constraints to system
    AffineConstraints<double> constraints;
    constraints.add_line_constraints(constraint_values);
    constraints.close();
    
    SparseMatrix<double> system_matrix;
    constraints.distribute_to_matrix(system_matrix, free_matrix);
    Vector<double> system_rhs;
    constraints.distribute_to_vector(system_rhs, free_rhs);
    
    // Solve
    Vector<double> solution(n_dofs);
    SolverControl solver_control(n_dofs * 10, 1e-12);
    SolverCG<double> solver(solver_control);
    PreconditionSSOR<double> preconditioner(system_matrix, 1.2);
    
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    
    if (solver_control.last_step() != 0)
    {
      std::cout << "Solver converged in " << solver_control.last_step() << " iterations" << std::endl;
    }
    
    // Compute consistent outward normal flux on interface
    // Flux = -k * grad(u) . n (our outward normal)
    // For consistent flux: residual of assembled system without BCs, divided by nodal weight
    
    // First, compute the residual with the constrained solution
    Vector<double> residual(n_dofs);
    free_matrix.vmult(residual, solution);
    residual -= free_rhs;
    
    // Now we need to extract interface nodes and compute flux
    // Interface is at x = iface_x
    std::vector<Point<2>> interface_points;
    std::vector<double> interface_values;
    std::vector<double> interface_fluxes;
    
    // Find all nodes on the interface
    for (unsigned int dofi = 0; dofi < n_dofs; ++dofi)
    {
      std::vector<Point<2>> node_points;
      std::vector<Tensor<1,2>> node_gradients;
      std::vector<double> node_weights;
      
      // Get all cells containing this node
      std::vector<const ActiveCellIterator<2>*> cells_with_node;
      for (auto &cell : dof_handler.active_cell_iterators())
      {
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
        {
          if (cell->vertex_index(v) == dofi)
          {
            cells_with_node.push_back(&cell);
            break;
          }
        }
      }
      
      if (cells_with_node.empty())
        continue;
      
      // Compute weighted average position
      Point<2> avg_pos(0);
      double total_weight = 0;
      for (auto *cell_ptr : cells_with_node)
      {
        auto &cell = *cell_ptr;
        unsigned int local_dofi = cell->vertex_index(dofi);
        double weight = cell->measure();
        avg_pos += cell->vertex(local_dofi) * weight;
        total_weight += weight;
      }
      if (total_weight > 0)
        avg_pos /= total_weight;
      
      // Check if node is on interface (x = iface_x)
      if (std::abs(avg_pos[0] - iface_x) < 1e-10)
      {
        interface_points.push_back(avg_pos);
        interface_values.push_back(solution[dofi]);
        
        // Compute consistent flux: residual / nodal weight
        // Nodal weight = sum of cell measures / 2 (for Q1)
        double nodal_weight = total_weight / 2.0;
        double flux = -residual[dofi] / nodal_weight;
        interface_fluxes.push_back(flux);
      }
    }
    
    // Write output file
    std::ofstream outfile("output.txt");
    outfile << interface_points.size() << std::endl;
    for (size_t i = 0; i < interface_points.size(); ++i)
    {
      outfile << interface_points[i][0] << " " << interface_points[i][1] << " "
              << interface_values[i] << " " << interface_fluxes[i] << std::endl;
    }
    outfile.close();
    
    std::cout << "Exported " << interface_points.size() << " interface points" << std::endl;
    
  }
  catch (ExceptionBase &exc)
  {
    exc.print_info(std::cerr);
    std::cerr << std::endl;
    return 1;
  }
  
  return 0;
}
