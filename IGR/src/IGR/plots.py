import torch
import numpy as np
import IGR.general as utils
from network import gradient
from skimage import measure
import trimesh


def plot_surface(decoder, scale, resolution, latent, mesh_level, unc_level):

    z, eikonal_losses, grid = compute_SDF_and_loss(resolution, latent, decoder, scale, unc_level)

    if mesh_level is not None:
        mesh = create_mesh(z, mesh_level, grid)
    else:
        mesh = None
    return mesh, eikonal_losses


def compute_SDF_and_loss(resolution, latent, decoder, scale, unc_level):
    grid = get_grid_uniform(resolution)

    z = []
    eikonal_losses = []

    for i, pnts in enumerate(torch.split(grid['grid_points'], 100000, dim=0)):

        pnts_cat = torch.cat((latent.expand(pnts.shape[0], -1), pnts), dim=1)

        z.append(decoder(pnts_cat).detach().cpu().numpy())
        mask = np.where(np.abs(z[-1]-unc_level*scale) < (1e-3*scale))[0]
        new_input = pnts_cat[mask]
        pnts_pred = decoder(new_input)
        pnts_grad = gradient(new_input, pnts_pred)
        eikonal_loss = (pnts_grad.norm(2, dim=-1) - 1).pow(2).detach().cpu().numpy()
        eikonal_losses.append(np.hstack((np.reshape(eikonal_loss, (-1, 1)), new_input[:, -3:].detach().cpu().numpy())))

    z = np.concatenate(z, axis=0)
    eikonal_losses = np.concatenate(eikonal_losses, axis=0)
    return z, eikonal_losses, grid


def create_mesh(z, level, grid):
    if (not (np.min(z) > level or np.max(z) < level)):
        z = z.astype(np.float64)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
        connected_comp = meshexport.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        meshexport = max_comp

    return meshexport


def get_grid_uniform(resolution):
    x = np.linspace(-1.1, 1.1, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = utils.to_cuda(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float,
                                             requires_grad=True))

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.2,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}