import autograd
import autograd.numpy as np
from .parallel_matmul import _par_matmul

@autograd.primitive
def matmul(a, b, threads=1):
    return _par_matmul(a, b, threads)
matmul.defgrad(lambda ans,a,b,threads=1: lambda g: matmul(g, np.transpose(b, (0,2,1)),
                                                          threads))
matmul.defgrad(lambda ans,a,b,threads=1: lambda g: matmul(np.transpose(a, (0,2,1)), g,
                                                          threads), argnum=1)

def einsum2(a, a_sublist, b, b_sublist, out_sublist):
    for subs in a_sublist, b_sublist, out_sublist:
        if len(subs) != len(set(subs)):
            raise NotImplementedError("Repeated subscripts not implemented")

    a, a_sublist = _sum_unique_axes(a, a_sublist, b_sublist, out_sublist)
    b, b_sublist = _sum_unique_axes(b, b_sublist, a_sublist, out_sublist)

    a_subs, b_subs, out_subs = map(set, (a_sublist, b_sublist, out_sublist))
    if out_subs - (a_subs | b_subs):
        raise ValueError("Output subscripts must be contained within input subscripts")

    a_minus_b = list(a_subs - b_subs)
    b_minus_a = list(b_subs - a_subs)
    # _sum_unique_axes should have removed any axes unique to a,b
    assert set(a_minus_b) <= out_subs and set(b_minus_a) <= out_subs

    ab = a_subs & b_subs
    abc = list(ab & out_subs)
    ab_minus_c = list(ab - out_subs)

    shapes = {}
    for arr,sublist in ((a,a_sublist), (b,b_sublist)):
        # arr.shape breaks in autograd if it has no dimension
        if sublist:
            for i,s in zip(arr.shape, sublist):
                if s not in shapes:
                    shapes[s] = i
                elif shapes[s] != i:
                    raise ValueError("a,b shapes don't match")

    c = matmul(_reshape(a, a_sublist, abc, a_minus_b, ab_minus_c),
               _reshape(b, b_sublist, abc, ab_minus_c, b_minus_a))

    c_sublist = abc + a_minus_b + b_minus_a
    c = np.reshape(c, [shapes[s] for s in c_sublist])

    return _transpose(c, c_sublist, out_sublist)

def _reshape(in_arr, in_sublist, *out_sublists):
    assert len(out_sublists) == 3

    old_sublist = in_sublist
    in_sublist = sum(out_sublists, [])
    in_arr = _transpose(in_arr, old_sublist, in_sublist)

    # in_arr.shape breaks in autograd if it has no dimension
    if in_sublist:
        shapes = {s:i for i,s in zip(in_arr.shape, in_sublist)}
    else: shapes = {}
    return np.reshape(in_arr, [np.prod([shapes[s] for s in out_subs], dtype=int)
                               for out_subs in out_sublists])

def _transpose(in_arr, in_sublist, out_sublist):
    if set(in_sublist) != set(out_sublist):
        raise ValueError("Input and output subscripts don't match")
    for sublist in (in_sublist, out_sublist):
        if len(set(sublist)) != len(sublist):
            raise NotImplementedError("Repeated subscripts not implemented")
    in_idxs = {k:v for v,k in enumerate(in_sublist)}
    return np.transpose(in_arr, axes=[in_idxs[s] for s in out_sublist])

def _sum_unique_axes(in_arr, in_sublist, *keep_subs):
    # assume no repeated subscripts
    assert len(in_sublist) == len(set(in_sublist))

    out_sublist = []
    sum_axes = []
    keep_subs = set([s for ks in keep_subs for s in ks])
    for idx, sub in enumerate(in_sublist):
        if sub in keep_subs:
            out_sublist.append(sub)
        else:
            sum_axes.append(idx)
    if sum_axes:
        return np.sum(in_arr, axis=tuple(sum_axes)), out_sublist
    else:
        return in_arr, out_sublist