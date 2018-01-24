from __future__ import division
from future.utils import with_metaclass
import numpy as np
import qinfer as qi
from abc import abstractmethod, abstractproperty, ABCMeta
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

################################################################################
# NODES
################################################################################

class NodeContext(object):
    def __init__(self):
        pass

class StructuredFilterNode(qi.SMCUpdater):
    
    def __init__(self, context):
        self._children = []
        self._child_weights = np.array([])
        self._parent = None
        self.context = context
        self.discovered_by_traversal = False
        self._master_node = None
        
        super(StructuredFilterNode, self).__init__(
                context.model, 1, context.prior, **context.updater_kwargs
            )

        
    @property
    def child_weights(self):
        """
        A list of weights of each child node.
        """
        return self._child_weights
    
    @property    
    def weight(self):
        """
        The weight of this node, as stored by its parent's `child_weights`.
        """
        if self.parent is None:
            return np.array(1)
        return np.asscalar(self.parent.child_weights[self.parent.get_child_idx(self)])
        
    @property
    def total_weight(self):
        """
        The product of weights descending from the `root_node` to 
        this node.
        """
        parent_weight = 1 if self.parent is None else self.parent.total_weight
        return np.asscalar(self.weight * parent_weight)
        
    def normalize_weights(self):
        """
        Normalizes `child_weights` to sum to 1.
        """
        total_weight = self.child_weights.sum()
        self._child_weights = self.child_weights / total_weight
        
    def reset_weights(self, new_weights=None):
        """
        Resets `child_weights` to a new value, by default, the uniform 
        distribution.
        
        :param list new_weights: The new weights to use, or `None` for 
            uniform weighting.
        """
        if new_weights is None:
            new_weights = np.ones(self.n_children) / self.n_children
        else:
            # peace of mind with all of the recursive functions floating around
            new_weights = np.atleast_1d(np.array(new_weights)).copy().flatten()
        self._child_weights = new_weights
        
    @property
    def children(self):
        """
        A list of `StructuredFilterNode`s.
        """
        return self._children
        
    @property
    def leaves(self):
        """
        A list of all leaves (nodes with no children) descended from this
        node. The order of this list is consistently left-to-right.
        """
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves += child.leaves
            return leaves
            
    @property
    def n_leaves(self):
        """
        The number of leaves (nes with no children descended from this node.
        """
        return len(self.leaves)
            
    @property
    def tree_depth(self):
        """
        The total depth of the tree that this node is a member of; the 
        depth of the deepest leaf descended from the `root_node`.
        """
        return max([leaf.depth for leaf in self.root_node.leaves])
        
    @property
    def parent(self):
        """
        The parent of this node.
        """
        return self._parent
        
    @property
    def branch_path(self):
        """
        A list specfying which children to pick, from the `root_node`, to 
        arrive at this node. For example, `[0,3,1]` specifies that
        this node is `root_node.children[0].children[3].children[1]`.
        """
        if self.parent is None:
            return []
        return self.parent.branch_path + [self.parent.get_child_idx(self)]
        
    def get_node_by_branch_path(self, branch_path):
        """
        Returns the node arrived at by following the given branch path from 
        the root node. It should hold that 
        `self is self.root_node.get_node_by_branch_path(self.branch_path)`.
        """
        if len(branch_path) == 0:
            return self
        return self.children[branch_path[0]].get_node_by_branch_path(branch_path[1:])
        
    def get_node_by_master(self, node):
        """
        Searches through the tree, starting at this node, looking for the 
        node whose `master_node` is the given node. If no such node is
        found, `None` is returned.
        """
        if self.master_node is node:
            return self
        for child in self.children:
            child_result = child.get_node_by_master(node)
            if child_result is not None:
                return child_result
        
    @property
    def root_node(self):
        """
        The root of the tree that this node is a member of; the node with 
        a `depth` of 0; the node with no `parent`.
        """
        return self if self.parent is None else self.parent.root_node
        
    def add_child(self, child, weight=1):
        """
        Appends a node as a child to this node.
        
        :param StructuredFilterNode child: The child node to append.
        :param float weight: The weight of this child relative to the 
            weights already present. Note that `normalize_weights` is 
            called after this value is appended to `child_weights`.
        """
        self._children.append(child)
        _weight = np.array([weight]).flatten()
        self._child_weights = np.concatenate([self._child_weights, _weight])
        self.normalize_weights()
        child._parent = self
    
    def get_child_idx(self, child):
        """
        Gets the index of the `children` list where the given child is 
        to be found. Raises `ValueError` if not found.
        
        :param StructuredFilterNode child: The child node to find.
        :returns int: The index of the given child.
        """
        for other_child_idx, other_child in enumerate(self.children):
            if child is other_child:
                return other_child_idx
        raise ValueError('No such child found.')
        
    def remove_child(self, child):
        """
        Removes the given child and its weight from this node.
        
        :param StructuredFilterNode child: The child node to remove.
        """
        child_idx = self.get_child_idx(child)
        del self._children[child_idx]
        self._child_weights = np.delete(self._child_weights, child_idx)
        child._parent = None
        self.normalize_weights()
        
    def replace_child(self, old_child, new_child):
        """
        Replaces one child of this node with another, keeping the weight intact.
        
        :param StructuredFilterNode old_child: The child to be replaced.
        :param StructuredFilterNode child: The node to replace it with.
        """
        old_child_idx = self.get_child_idx(old_child)
        self._children[old_child_idx] = new_child
        new_child._parent = self
        old_child._parent = None

    @property
    def n_children(self):
        """
        The number of children owned by this node.
        """
        return len(self.children)
    
    @property    
    def is_leaf(self):
        """
        Whether or not this node is a leaf of the tree it belongs to; boolean.
        """
        return self.n_children == 0
        
    def reset_discovered_flags(self):
        """
        Sets the property `discovered_by_traversal` to `False` of this node,
        and all nodes descended from this node.
        """
        self.discovered_by_traversal = False
        for child in self.children:
            child.reset_discovered_flags()

    @property
    def depth(self):
        """
        The depth of this node in the tree it belongs to, where the `root_node`
        has depth 0.
        """
        if self.parent is None:
            return 0
        return self.parent.depth + 1
    
    @property
    def master_node(self):
        """
        If this node was copied from another node (with `copy_node`), the 
        original instance is returned. Otherwise, `None` is returned.
        """
        return self._master_node
        
    @property
    def tree_size(self):
        """
        The total number of nodes in the tree whose root node is this one.
        """
        return 1 + sum([child.tree_size for child in self.children])
    
    def copy_node(self, node_class=None):
        """
        Returns a copy of this node; a new instance
        of the same class with the same context. Links to children and parent are
        discarded. Particle weights and particle locations are set on the
        new copy if and only if this node is a leaf. Note that these 
        arrays are not copied, being references to the same spot in memory.

        :returns: An instance of some subclass of `StructuredFilterNode`.
        """
        new_node = type(self)(self.context)
        if self.is_leaf:
            new_node.particle_weights = self.particle_weights
            new_node.particle_locations = self.particle_locations
        new_node._master_node = self if self.master_node is None else self.master_node
        return new_node
        
    def copy_tree(self):
        """
        Returns a new tree where a copy of this node is the `root_node`. This
        is done using recursion and the `copy_node` method.
        
        :returns: An instance of some subclass of `StructuredFilterNode`.
        """
        new_root = self.copy_node()
        if self.is_leaf:
            return new_root
        for child in self.children:
            new_root.add_child(child.copy_tree())
        new_root.reset_weights(self.child_weights)
        return new_root
        
    @property
    def x_position(self):
        """
        The x-position of this node in a planar embedding of the tree it
        belongs to.
        """
        return 0
        
    @property
    def y_position(self):
        """
        The y-position of this node in a planar embedding of the tree it
        belongs to.
        """
        return self.tree_depth - self.depth
        
    def plot_posterior_marginal(self, idx_param0=0, idx_param1=None, **kwargs):
        if idx_param1 is None:
            return super(StructuredFilterNode, self).plot_posterior_marginal(
                idx_param0, **kwargs
            )
            
        if not kwargs.has_key('alpha'):
            kwargs['alpha'] = 10 / self.n_particles
        
        plt.sca(plt.gca())    
        plt.scatter(
            self.particle_locations[:,idx_param0],
            self.particle_locations[:,idx_param1],
            s = self.particle_weights / self.particle_weights.max(),
            **kwargs
        )
        
    def plot_tree(self, axis=None):
        """
        Draws a planar embedding of the tree descended from this node on
        a `matplotlib` axis.
        """
        graphics = GraphicsNodeOperation()
        traverser = DepthFirstTreeTraversal()
        traverser.add_node_operation(graphics)
        traverser(self)
        
        if axis is None:
            axis = plt.gca()
        
        axis.set_aspect(1)
        axis.axis('off')
        axis.set_xlim([0, 2 * graphics.x_offset + self.n_leaves * graphics.x_spacing])
        axis.set_ylim([1 + (self.tree_depth + 2) * graphics.y_spacing, 1])
        
        for patch in graphics.graphics_list:
            axis.add_artist(patch)
        
        
class StructuredFilterLeaf(StructuredFilterNode):
    """
    A class that specializes `StructuredFilterLeaf` to those nodes which 
    will certainly be leaves.
    """
    
    def add_child(self, child):
        raise AttributeError('This node is a leaf and cannot be assigned children.')
        
    @property
    def x_position(self):
        all_leaves = self.root_node.leaves
        for idx_leaf, leaf in enumerate(all_leaves):
            if leaf is self:
                return idx_leaf
    
class StructuredFilterParent(StructuredFilterNode):
    """
    A class that specializes `StructuredFilterNode` to those nodes which 
    will certainly have children.
    """
    
    @property
    def x_position(self):
        return (self.children[0].x_position + self.children[-1].x_position) / 2
        
    @property
    def n_particles(self):
        if self.is_leaf:
            return 1
        else:
            return sum([child.n_particles for child in self.children])
    
    @property
    def particle_locations(self):
        if self.is_leaf:
            return np.empty((1, self.model.n_modelparams))
        else:
            return np.concatenate(
                [child.particle_locations for child in self.children],
                axis=0
            )
    @particle_locations.setter
    def particle_locations(self, locs):
        if not self.is_leaf and locs.shape[0] != self.n_particles:
            raise ValueError('When setting particle_locations for a non-leaf node, you cannot change the shape.')
        idx = 0
        for child, weight in zip(self.children, self.child_weights):
            child.particle_locations = locs[idx:idx+child.n_particles,:]
            idx += child.n_particles
            
    @property
    def particle_weights(self):
        if self.is_leaf:
            return np.ones((1,))
        else:
            return np.concatenate(
                [
                    weight * child.particle_weights 
                    for child, weight in zip(self.children, self.child_weights)
                ],
                axis=0
            )
    @particle_weights.setter
    def particle_weights(self, weights):
        if not self. is_leaf and weights.size != self.n_particles:
            raise ValueError('When setting particle_weights for a non-leaf node, you cannot change the shape.')
        idx = 0
        for child, weight in zip(self.children, self.child_weights):
            child.particle_weights = weights[idx:idx+child.n_particles] / weight
            idx += child.n_particles

class ModelSelectorNode(StructuredFilterParent):
    pass
        

class MixtureNode(StructuredFilterParent):
    pass
            
    
class ParticleFilterNode(StructuredFilterLeaf):
    def __init__(self, context, n_particles=1):
        super(ParticleFilterNode, self).__init__(context)
        self._resampler = self.resampler
        self.resample(n_particles=n_particles)      

    def _maybe_resample(self, **kwargs):
        ess = self.n_ess
        if ess <= 10:
            warnings.warn(
                "Extremely small n_ess encountered ({}). "
                "Resampling is likely to fail. Consider adding particles, or "
                "resampling more often.".format(ess),
                ApproximationWarning
            )
        if ess < self.n_particles * self.context.resample_thresh:
            self.resample()
            pass
        
    def resample(self, **kwargs):
        # the resampler in SMCUpdater is poorly implemented at the moment,
        # and we fudge the ability to pass extra arguments to the underlying
        # resampler here
        
        # trick self.resample into thinking this function is a qi.Resampler.
        def resampler_call(model, particle_dist):
            return self._resampler(model, particle_dist, **kwargs)
        self.resampler = resampler_call
        super(ParticleFilterNode, self).resample()
        self.resampler = self._resampler


################################################################################
# NODE OPERATIONS
################################################################################

class NodeOperation(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __call__(self, node):
        pass
        
class DiscriminatingNodeOperation(NodeOperation):
    def __call__(self, node, *args):
        result = {}
        if isinstance(node, StructuredFilterParent):
            result['StructuredFilterParent'] = self.parent_node_operation(node, *args)
        if isinstance(node, ModelSelectorNode):
            result['ModelSelectorNode'] = self.model_selector_node_operation(node, *args)
        if isinstance(node, MixtureNode):
            result['MixtureNode'] = self.mixture_node_operation(node, *args)
        if isinstance(node, ParticleFilterNode):
            result['ParticleFilterNode'] = self.particle_filter_node_operation(node, *args)
        return result
            
    def model_selector_node_operation(self, node, *args):
        pass
    def mixture_node_operation(self, node, *args):
        pass
    def particle_filter_node_operation(self, node, *args):
        pass
    def parent_node_operation(self, node, *args):
        pass
        
class DecisionTreeRule(DiscriminatingNodeOperation):
    def __init__(self):
        self.tree_list = []
        self.weight_list = np.array([])
        
    @property
    def n_trees(self):
        return len(self.tree_list)
        
    def parent_node_operation(self, node):
        if node.parent is None:
            # we are at the root node, start fresh
            self.tree_list = [node.copy_node()]
            self.weight_list = np.array([1])
            
    def model_selector_node_operation(self, node):
        new_tree_list = []
        new_weight_list = np.array([])
        for tree_weight, tree in zip(self.weight_list, self.tree_list):
            if tree.get_node_by_master(node) is None:
                # the current node doesn't figure in the tree under 
                # consideration, so just let it be.
                new_tree_list.append(tree.copy_tree())
                new_weight_list = np.concatenate([
                        new_weight_list, 
                        np.array([tree_weight])
                    ])
            else:
                # otherwise, we add a new tree for every possibility we
                # could select
                for child in node.children:
                    new_tree = tree.copy_tree()
                    node_in_new_tree = new_tree.get_node_by_master(node)
                    node_in_new_tree.add_child(child.copy_node())
                    
                    new_tree_list.append(new_tree)
                    new_weight_list = np.concatenate([
                            new_weight_list, 
                            np.atleast_1d([tree_weight * child.weight]).flatten()
                        ])
                
        new_order = np.argsort(new_weight_list)[::-1]
        self.tree_list = [new_tree_list[idx] for idx in new_order]
        self.weight_list = new_weight_list[new_order]
        
    def mixture_node_operation(self, node):
        for tree_weight, tree in zip(self.weight_list, self.tree_list):
            node_in_new_tree = tree.get_node_by_master(node)
            if node_in_new_tree is not None:
                for child in node.children:
                    node_in_new_tree.add_child(child.copy_node())
                node_in_new_tree.reset_weights(node.child_weights)
                
        
################################################################################
# TRAVERSING TREES OF NODES
################################################################################

class TreeTraversal(with_metaclass(ABCMeta, object)):
    def __init__(self):
        self._node_operations = []
    
    @property
    def node_operations(self):
        return self._node_operations
        
    def add_node_operation(self, node_operation):
        self._node_operations.append(node_operation)
        
    @abstractmethod
    def node_iterator(self, root_node):
        pass
    
    def __call__(self, root_node):
        for node in self.node_iterator(root_node):
            for node_operation in self.node_operations:
                node_operation(node)
        
class DepthFirstTreeTraversal(TreeTraversal):
                        
    def node_iterator(self, node, first_call=True):
        # recursive implementation of depth-first search from wikipedia
        yield node
        if first_call:
            node.reset_discovered_flags()
        node.discovered_by_traversal = True
        for child in node.children:
            if not child.discovered_by_traversal:
                # this superfluous for loop is necessary in python 2
                for x in self.node_iterator(child, first_call=False):
                    yield x
                
class BreadthFirstTreeTraversal(TreeTraversal):
    
    def node_iterator(self, node, first_call=True):
        yield node
        if first_call:
            node.reset_discovered_flags()
        # there is no need in a breadth-first traversal to use flags, but
        # it is helpful for debugging to make sure all nodes are visited.
        node.discovered_by_traversal = True
        for child in node.children:
            # this superfluous for loop is necessary in python 2
            for x in self.node_iterator(child, first_call=False):
                yield x
                
class ChildRespondingTreeTraversal(TreeTraversal):
    """
    Represents a tree-traversal where node operations recieve, as input, 
    both the node in question and the results of each of their children.
    """
        
    def node_iterator(self, node):
        raise RuntimeError('This traversal does not use the node iterator pardigm.')
        
    def parse_node_operation_results(self, node_operation_results):
        return node_operation_results[0]

    def __call__(self, root_node, first_call=True):
        if first_call:
            root_node.reset_discovered_flags()
        root_node.discovered_by_traversal = True
            
        # recursively collect results from children
        child_results = [
            self(node, first_call=False) for node in root_node.children
        ]
        
        # feed these results to each node_operation
        node_operation_results = [
            node_operation(root_node, child_results)
            for node_operation in self.node_operations
        ]
        
        # somehow parse these results into one output
        return self.parse_node_operation_results(node_operation_results)
            
            
################################################################################
# CLUSTERING FUNCTIONS
################################################################################

def centroid_distances(centroids, particle_locations):
    """
    Returns the squared Euclidean distance between all pairs of 
    `particle_locations` and `centroids`.
    
    :param np.ndarray particle_locations: Shape `(n_particles, n_mps)`.
    :param np.ndarray centroids: Shape `(n_centroids, n_mps)`.
    
    :return np.ndarray: Shape `(n_centroids, n_particles)`.
    """
    return np.sum(np.abs(
            particle_locations[np.newaxis,:,:] - 
            centroids[:,np.newaxis,:]
        ) ** 2, axis=2)

def kmeans_initializer(particle_locations, n_clusters):
    """
    Returns a guess of cluster centroids for the given particle locations 
    based on the K-Means++ heuristic.
    
    :param np.ndarray particle_locations: Shape `(n_particles, n_mps)`.
    :param int n_clusters: How many clusters to divide the particles into.
    
    :return np.ndarray: Shape `(n_clusters, n_mps)`.
    """
    n_particles = particle_locations.shape[0]
    n_mps = particle_locations.shape[1]
    centroids = np.empty((n_clusters, n_mps))
    centroids[0,:] = particle_locations[np.random.choice(n_particles), :]
    for idx_centroid in range(1, n_clusters):
        current_centroids = centroids[:idx_centroid, :]
        distances = np.amin(
                centroid_distances(current_centroids, particle_locations),
                axis=0
            )
        distances /= distances.sum()
        new_idx = np.random.choice(n_particles, p=distances)
        centroids[idx_centroid, :] = particle_locations[new_idx, :]
    return centroids
    
def weighted_kmeans(particle_weights, particle_locations, n_clusters, max_iterations=500):
    n_particles = particle_locations.shape[0]
    n_mps = particle_locations.shape[1]
    centroids = kmeans_initializer(particle_locations, n_clusters)
    labels = np.argmin(centroid_distances(centroids, particle_locations), axis=0)
    
    for idx_iter in range(max_iterations):
        mask = labels[np.newaxis,:] == np.arange(n_clusters)[:,np.newaxis]
        centroids = np.sum(
                mask[:,:,np.newaxis] * 
                    particle_weights[np.newaxis,:,np.newaxis] * 
                    particle_locations[np.newaxis,:,:],
                axis=1
            )
        centroids /= np.sum(mask * particle_weights[np.newaxis:], axis=1)[:,np.newaxis]
        new_labels = np.argmin(centroid_distances(centroids, particle_locations), axis=0)
        if np.array_equal(labels, new_labels):
            return labels, centroids
        labels = new_labels
        
    raise RuntimeError('K-Means did not converge after {} iterations'.format(max_iterations))

################################################################################
#  PLOTTING AND GRAPHICS
################################################################################

class GraphicsNodeOperation(DiscriminatingNodeOperation):
    def __init__(self, radius=0.02, 
            x_offset=0.1, y_offset=0.9, x_spacing=.1, y_spacing=-.1,
            circle_color='lightpink', 
            square_color='orange', 
            triangle_color='deepskyblue',
            line_color='k'
        ):
        self.radius = radius
        self.circle_color = circle_color
        self.square_color = square_color
        self.triangle_color = triangle_color
        self.line_color=line_color
        
        self.x_offset, self.y_offset = x_offset, y_offset
        self.x_spacing, self.y_spacing = x_spacing, y_spacing
        
        self.graphics_list = []
        
    def location(self, node):
        return [
                self.x_offset + self.x_spacing * node.x_position, 
                self.y_offset + self.y_spacing * node.depth
            ]
            
    def model_selector_node_operation(self, node):
        self.graphics_list.append(
                mpl.patches.Circle(
                    self.location(node), self.radius, color=self.circle_color,
                    zorder=1e6
                )
            )
        
    def mixture_node_operation(self, node):
        loc = self.location(node)
        angles = [np.pi/2, -np.pi/6, -5*np.pi/6]
        xy = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        xy = np.array(loc) + 1.4 * self.radius * xy
        self.graphics_list.append(mpl.patches.Polygon(xy, 
            facecolor=self.triangle_color, edgecolor=self.triangle_color,
            zorder=1e6
        ))
        
    def particle_filter_node_operation(self, node):
        loc = self.location(node)
        loc[0] = loc[0] - self.radius
        loc[1] = loc[1] - self.radius
        self.graphics_list.append(
                mpl.patches.Rectangle(
                    loc, 2*self.radius, 2*self.radius, 
                    edgecolor=self.square_color, facecolor=self.square_color,
                    zorder=1e6
                )
            )
            
    def parent_node_operation(self, node):
        node_loc = self.location(node)
        for child in node.children:
            width = child.weight * self.radius * 100
            child_loc = self.location(child)
            self.graphics_list.append(
                mpl.lines.Line2D(
                    [node_loc[0], child_loc[0]], [node_loc[1], child_loc[1]],
                    color=self.line_color, linewidth=width
                )
            )

################################################################################
# IMPLEMENTATION OF GRANADE AND WIEBE's STRUCTURED FILTERING
################################################################################

class ChampionPruningRule(DiscriminatingNodeOperation):
    def parent_node_operation(self, node):
        if node.n_children > 1:
            strongest_child_idx = int(np.argmax(node.child_weights))
            strongest_child = node.children[strongest_child_idx]
            weight = node.child_weights[strongest_child_idx]
            if weight / (1 - weight) >= node.context.champion_bayes_factor:
                for child in node.children:
                    if child is not strongest_child:
                        node.remove_child(child)
                    
class FloorPruningRule(DiscriminatingNodeOperation):
    def model_selector_node_operation(self, node):
        for child, weight in zip(node.children, node.child_weights):
            if weight < node.context.floor_weight:
                node.remove_child(child)
                
class OnlyChildPruningRule(DiscriminatingNodeOperation):
    def parent_node_operation(self, node):
        parent = node.parent
        if parent is not None and parent.n_children == 1:
            for child in node.children:
                parent.add_child(child)
            parent.remove_child(node)
            parent.reset_weights(node.child_weights)
                
class SingleChildPruningRule(DiscriminatingNodeOperation):
    def parent_node_operation(self, node):
        if node.parent is not None and node.n_children == 1:
            child = node.children[0]
            node.parent.replace_child(node, child)
                
class SplittingRule(DiscriminatingNodeOperation):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
    def particle_filter_node_operation(self, node):
        model_selector_node = node.context.new_model_selector_node()
        node.parent.replace_child(node, model_selector_node)
        
        for n_clusters in node.context.n_clusters_list:
            if n_clusters == 1:
                model_selector_node.add_child(node)
                if not self._kwargs.has_key('n_particles'):
                    node.resample(n_particles=node.n_particles, **self._kwargs)
                else:
                    node.resample(**self._kwargs)
            else:
                cluster_idxs, _ = node.context.clusterer(
                        node.particle_weights, 
                        node.particle_locations, 
                        n_clusters
                    )
                mixture_node = node.context.new_mixture_node()
                for n_cluster in range(n_clusters):
                    filter_node = node.context.new_particle_filter_node()
                    mixture_node.add_child(filter_node)
                    
                    mask = cluster_idxs == n_cluster
                    filter_node.particle_locations = node.particle_locations[mask, :]
                    filter_node.particle_weights = node.particle_weights[mask]
                    filter_node.particle_weights /= filter_node.particle_weights.sum()
                    
                    n_particles = max(filter_node.particle_weights.size, node.context.min_n_particles)
                    filter_node.resample(
                            n_particles = n_particles,
                            **self._kwargs
                        )
                mixture_node.reset_weights()
                model_selector_node.add_child(mixture_node)
        model_selector_node.reset_weights()
                
class ResamplingRule(DiscriminatingNodeOperation):
    def __init__(self, splitting_rule):
        self.splitting_rule = splitting_rule
        
    def particle_filter_node_operation(self, node):
        if node.n_ess < node.n_particles * node.resample_thresh:
            if node.depth < node.context.max_depth:
                self.splitting_rule(node)
                
        
class UpdateRule(DiscriminatingNodeOperation):
    def __init__(self, outcome, expparams, check_for_resample=True):
        self.outcome = outcome
        self.expparams = expparams
        self.check_for_resample = check_for_resample
            
    def particle_filter_node_operation(self, node, child_results):
        # do a basic update
        try:
            node.update(self.outcome, self.expparams, check_for_resample=False)
        except RuntimeError:
            # All weights are zero. We want to delete it, but that will
            # mess up the recursion. So we set its weight to 0, and pruning
            # will deal with it later.
            parent_weights = node.parent.child_weights
            parent_weights[node.parent.get_child_idx(node)] = 0
            node.parent.reset_weights(parent_weights)
            node.parent.normalize_weights()
            return 0
        # the total likelihood of the data give the prior is the 
        # last member of the normalization record.
        return node.normalization_record[-1]
        
    def parent_node_operation(self, node, child_results):
        new_child_weights = node.child_weights * np.array(child_results).flatten()
        node.reset_weights(new_child_weights)
        node.normalize_weights()
        return new_child_weights.sum()
            
    def __call__(self, node, child_results):
        result = super(UpdateRule, self).__call__(node, child_results)
        # we want to always return the total likelihood
        if result.has_key('StructuredFilterParent'):
            return result['StructuredFilterParent']
        return result['ParticleFilterNode']

class GranadeWiebeContext(NodeContext):
    def __init__(self, 
            model, initial_n_particles, prior,
            clusterer=weighted_kmeans,
            model_selector_node_class=ModelSelectorNode,
            mixture_node_class=MixtureNode,
            particle_filter_node_class=ParticleFilterNode,
            **updater_kwargs
        ):
        self.max_depth = 4
        self.n_clusters_list = [1, 2]
        self.floor_weight = 0.1
        self.champion_bayes_factor = 2000
        self.min_n_particles = 1000
        self.resample_thresh = 0.5
        
        self.clusterer = weighted_kmeans
        
        self.model_selector_node_class = model_selector_node_class
        self.mixture_node_class = mixture_node_class
        self.particle_filter_node_class = particle_filter_node_class
        
        self.model = model
        self.initial_n_particles = max(initial_n_particles, self.min_n_particles)
        self.prior = prior
        self.updater_kwargs = updater_kwargs
        
    def new_model_selector_node(self):
        return self.model_selector_node_class(self)
        
    def new_mixture_node(self):
        return self.mixture_node_class(self)
        
    def new_particle_filter_node(self, n_particles=1):
        particle_filter_node = self.particle_filter_node_class(self)
        particle_filter_node.reset(n_particles)
        return particle_filter_node

class StructuredFilter(ModelSelectorNode):
    def __init__(self, context, make_initial_filter=True):
        self.context = context
        super(StructuredFilter, self).__init__(context)
        if make_initial_filter:
            self.reset()
        else:
            for child in self.children:
                self.remove_child(child)
        
    def copy_node(self):
        new_node = type(self)(self.context, make_initial_filter=False)
        new_node._master_node = self if self.master_node is None else self.master_node
        return new_node
        
    def reset(self, n_particles=None):
        for child in self.children:
            self.remove_child(child)
        if n_particles is None:
            n_particles = self.context.initial_n_particles
        self.add_child(
            self.context.new_particle_filter_node(n_particles)
        )
        
    def get_structured_particle_distribution(self):
        traversal = BreadthFirstTreeTraversal()
        decision_tree = DecisionTreeRule()
        traversal.add_node_operation(decision_tree)
        # perform the traversal on self; the root node
        traversal(self)
        
        tree_weights = decision_tree.weight_list
        trees = decision_tree.tree_list
        
        particle_locations_list = [tree.particle_locations for tree in decision_tree.tree_list]
        particle_weights_list = [tree.particle_weights for tree in decision_tree.tree_list]
        
        return tree_weights, particle_weights_list, particle_locations_list
        
    def plot_posterior_marginal(self, idx_param0=0, idx_param1=None, **kwargs):
        
        traversal = BreadthFirstTreeTraversal()
        decision_tree = DecisionTreeRule()
        traversal.add_node_operation(decision_tree)
        # perform the traversal on self; the root node
        traversal(self)
        
        for tree_weight, tree in zip(decision_tree.weight_list, decision_tree.tree_list):
            super(StructuredFilter, tree).plot_posterior_marginal(
                idx_param0, idx_param1, **kwargs
            )
        
        
    def prune(self):
        traversal = BreadthFirstTreeTraversal()
        traversal.add_node_operation(ChampionPruningRule())
        traversal.add_node_operation(FloorPruningRule())
        traversal.add_node_operation(OnlyChildPruningRule())
        traversal.add_node_operation(SingleChildPruningRule())
        # perform the traversal on self; the root node
        traversal(self)
        
    def resample(self, **kwargs):
        traversal = BreadthFirstTreeTraversal()
        traversal.add_node_operation(ResamplingRule(SplittingRule(**kwargs)))
        # perform the traversal on self; the root node
        traversal(self)
        
    def update(self, outcome, expparam, check_for_resample=True):
        
        self._data_record.append(outcome)
        self._just_resampled = False
        
        udpate_traversal = ChildRespondingTreeTraversal()
        udpate_traversal.add_node_operation(
            UpdateRule(outcome, expparam, check_for_resample=check_for_resample)
        )
        # update all weights: perform the traversal on self; the root node
        norm = udpate_traversal(self)
        
        # the traversal conveniently returns the total likelihood of the given outcome
        self._normalization_record.append(norm)
        
        # Update the particle locations according to the model's timestep.
        self.particle_locations = self.model.update_timestep(
            self.particle_locations, expparam
        )[:, :, 0]

        # Check if we need to update our min_n_ess attribute.
        if self.n_ess <= self._min_n_ess:
            self._min_n_ess = self.n_ess
        
        # prune the tree, getting rid of useless branches
        self.prune()
        
        if check_for_resample:
            self.resample()

        
        
        
        
