#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/numbers.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <cmath>

using namespace dealii;

class SourceTerm : public Function<2> {
public:
    virtual double value(const Point<2> &p, const unsigned int = 0) const override {
        return 5.0 * numbers::PI * numbers::PI * (1.4 - p[0]) * std::cos(numbers::PI * p[1]);
    }
};

int main() {
    try {
        // ── Read input ──
        std::ifstream infile("input.txt");
        if (!infile.is_open()) {
            std::cerr << "Cannot open input.txt\n";
            return 1;
        }

        double x0, x1, y0, y1, iface_x, k;
        int nx, ny;
        infile >> x0 >> x1 >> y0 >> y1 >> iface_x >> k >> nx >> ny;

        size_t n_flux;
        infile >> n_flux;
        std::vector<double> y_flux(n_flux), q_flux(n_flux);
        for (size_t i = 0; i < n_flux; ++i) {
            infile >> y_flux[i] >> q_flux[i];
        }
        infile.close();

        // Sort flux data by y for interpolation
        {
            std::vector<size_t> idx(n_flux);
            for (size_t i = 0; i < n_flux; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                      [&](size_t a, size_t b) { return y_flux[a] < y_flux[b]; });
            std::vector<double> y_sorted(n_flux), q_sorted(n_flux);
            for (size_t i = 0; i < n_flux; ++i) {
                y_sorted[i] = y_flux[idx[i]];
                q_sorted[i] = q_flux[idx[i]];
            }
            y_flux = y_sorted;
            q_flux = q_sorted;
        }

        // ── Create mesh: nx × ny elements ──
        Triangulation<2> tria;
        GridGenerator::subdivided_hyper_rectangle(tria, {nx, ny},
                                                   Point<2>(x0, y0),
                                                   Point<2>(x1, y1));

        // ── FE space Q1 ──
        FE_Q<2> fe(1);
        DoFHandler<2> dof_handler(tria);
        dof_handler.distribute_dofs(fe);
        const unsigned int n_dofs = dof_handler.n_dofs();

        // Build vertex index to DOF index map (Q1: one DOF per vertex)
        std::map<unsigned int, unsigned int> vertex_to_dof;
        for (const auto &cell : dof_handler.active_cell_iterators()) {
            std::vector<unsigned int> dof_indices;
            cell->get_dof_indices(dof_indices);
            for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v) {
                vertex_to_dof[cell->vertex_index(v)] = dof_indices[v];
            }
        }

        // ── Sparsity pattern ──
        DynamicSparsityPattern dsp(n_dofs, n_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        dsp.compress();
        SparseMatrix<double> A_free;  // matrix WITHOUT any BC applied
        A_free.reinit(dsp);

        Vector<double> b_volume(n_dofs);
        b_volume = 0.0;

        // ── Assemble volume terms ──
        QGauss<2> q_cell(3);
        SourceTerm source;

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            FEValues<2> fe_values(fe, *cell,
                update_values | update_gradients | update_quadrature_points | update_JxW_values,
                q_cell);

            const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
            FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
            Vector<double> cell_rhs(dofs_per_cell);

            for (unsigned int q = 0; q < q_cell.size(); ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        cell_matrix(i, j) += k * (fe_values.shape_grad(i, q) *
                                                  fe_values.shape_grad(j, q)) *
                                             fe_values.JxW(q);
                    }
                    cell_rhs(i) += source.value(fe_values.quadrature_point(q)) *
                                   fe_values.shape_value(i, q) * fe_values.JxW(q);
                }
            }

            const std::vector<unsigned int> &dof_indices = cell->get_dof_indices();
            A_free.add(dof_indices, cell_matrix);
            b_volume.add(dof_indices, cell_rhs);
        }

        // ── Assemble interface Neumann term ──
        QGauss<1> q_face(2);

        for (const auto &face : dof_handler.active_face_iterators()) {
            if (face->at_boundary()) {
                const Point<2> center = face->center();
                if (std::abs(center[0] - iface_x) < 1e-12) {
                    const auto &cell = *face->cell();
                    const unsigned int face_no = face->face_no();

                    FEFaceValues<2> fe_face_values(fe, cell, face_no,
                        update_values | update_quadrature_points | update_JxW_values,
                        q_face);

                    const unsigned int dofs_per_face = fe_face_values.n_dofs_per_face();
                    Vector<double> face_rhs(dofs_per_face);

                    for (unsigned int q = 0; q < q_face.size(); ++q) {
                        const Point<2> qp = fe_face_values.quadrature_point(q);
                        // Linear interpolation of imported flux
                        double g = 0.0;
                        if (n_flux > 0) {
                            if (qp[1] <= y_flux.front())
                                g = q_flux.front();
                            else if (qp[1] >= y_flux.back())
                                g = q_flux.back();
                            else {
                                for (size_t i = 0; i < n_flux - 1; ++i) {
                                    if (qp[1] >= y_flux[i] - 1e-15 &&
                                        qp[1] <= y_flux[i + 1] + 1e-15) {
                                        double t = (qp[1] - y_flux[i]) /
                                                    (y_flux[i + 1] - y_flux[i]);
                                        g = q_flux[i] * (1.0 - t) + q_flux[i + 1] * t;
                                        break;
                                    }
                                }
                            }
                        }
                        for (unsigned int i = 0; i < dofs_per_face; ++i) {
                            face_rhs(i) += g * fe_face_values.shape_value(i, q) *
                                           fe_face_values.JxW(q);
                        }
                    }

                    std::vector<unsigned int> face_dof_indices;
                    face->get_dof_indices(face_dof_indices);
                    b_volume.add(face_dof_indices, face_rhs);
                }
            }
        }

        // ── Dirichlet BC: u = 0 at x = x1 ──
        std::map<types::global_dof_index, double> dirichlet_values;
        for (const auto &vertex : tria.active_vertex_iterators()) {
            if (std::abs(vertex->value()[0] - x1) < 1e-12) {
                auto it = vertex_to_dof.find(vertex->index());
                if (it != vertex_to_dof.end()) {
                    dirichlet_values[it->second] = 0.0;
                }
            }
        }

        SparseMatrix<double> A = A_free;
        Vector<double> b = b_volume;
        MatrixTools::apply_boundary_values(dirichlet_values, A, b);

        // ── Solve ──
        SolverControl sc(2000, 1e-14);
        PreconditionSOR prec;
        prec.initialize(A, PreconditionSOR::Properties(1.2));
        SolverCG<> solver(sc);
        Vector<double> solution(n_dofs);
        solver.solve(A, solution, b, prec);

        // ── Identify interface nodes ──
        std::set<unsigned int> iface_vertex_set;
        for (const auto &face : dof_handler.active_face_iterators()) {
            if (face->at_boundary()) {
                const Point<2> center = face->center();
                if (std::abs(center[0] - iface_x) < 1e-12) {
                    for (unsigned int v = 0; v < GeometryInfo<2>::faces_per_cell; ++v) {
                        iface_vertex_set.insert(face->vertex(v)->index());
                    }
                }
            }
        }

        // Collect (y, dof_index) for interface nodes, sort by y
        std::vector<std::pair<double, unsigned int>> iface_nodes;
        for (unsigned int vidx : iface_vertex_set) {
            const Point<2> &pt = tria.vertex(vidx)->value();
            auto it = vertex_to_dof.find(vidx);
            if (it != vertex_to_dof.end()) {
                iface_nodes.emplace_back(pt[1], it->second);
            }
        }
        std::sort(iface_nodes.begin(), iface_nodes.end());

        // ── Solution values at interface nodes ──
        std::vector<double> iface_values;
        std::vector<Point<2>> iface_coords;
        for (auto &p : iface_nodes) {
            for (const auto &vertex : tria.active_vertex_iterators()) {
                if (vertex->index() == vertex_to_dof.find(p.second)->first) {
                    iface_coords.push_back(vertex->value());
                    break;
                }
            }
            iface_values.push_back(solution(p.second));
        }

        // ── Consistent flux recovery: q = -(A_free*u - b_volume) / h_trib ──
        Vector<double> residual(n_dofs);
        A_free.vmult(residual, solution);
        residual -= b_volume;

        const double hy = (y1 - y0) / ny;
        const double h_trib = hy;  // tributary length along interface for interior nodes

        std::vector<double> iface_fluxes;
        for (size_t i = 0; i < iface_nodes.size(); ++i) {
            double h = h_trib;
            // Corner nodes: half tributary length
            if (std::abs(iface_coords[i][1] - y0) < 1e-12 ||
                std::abs(iface_coords[i][1] - y1) < 1e-12) {
                h = h_trib * 0.5;
            }
            iface_fluxes.push_back(-residual(iface_nodes[i].second) / h);
        }

        // ── Write output ──
        std::ofstream outfile("output.txt");
        outfile << iface_nodes.size() << "\n";
        for (size_t i = 0; i < iface_nodes.size(); ++i) {
            outfile << iface_coords[i][0] << " " << iface_coords[i][1] << " "
                    << iface_values[i] << " " << iface_fluxes[i] << "\n";
        }
        outfile.close();

        std::cout << "Interface nodes: " << iface_nodes.size() << "\n";
        std::cout << "DOF count: " << n_dofs << "\n";
        std::cout << "Solver iterations: " << sc.last_step() << "\n";
        std::cout << "Solver tolerance: " << sc.last_value() << "\n";

    } catch (std::exception &exc) {
        std::cerr << "Exception: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
