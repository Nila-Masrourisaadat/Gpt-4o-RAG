import re
import warnings

def text_split_fuzzy(text: str,
        chunk_size: int,
        chunk_overlap: int=0,
        separator: str='\n\n',
        joiner=None,
        len_func=len
    ):

    assert separator, 'Separator must be non-empty'

    if ((not isinstance(text, str))
        or (not isinstance(separator, str))):
        raise ValueError(f'text and separator must be strings.\n'
                         f'Got {text.__class__} for text and {separator.__class__} for separator')

    if chunk_overlap == 0:
        msg = 'chunk_overlap must be a positive integer. For no overlap, use text_split() instead'
        raise ValueError(msg)

    if ((not isinstance(chunk_size, int))
        or (not isinstance(chunk_overlap, int))
        or (chunk_size <= 0)
        or (chunk_overlap < 0)
        or (chunk_size < chunk_overlap)):
        raise ValueError(f'chunk_size must be a positive integer, '
                         f'chunk_overlap must be a non-negative integer, and'
                         f'chunk_size must be greater than chunk_overlap.\n'
                         f'Got {chunk_size} chunk_size and {chunk_overlap} chunk_overlap.')

    # Split up the text by the separator
    sep_pat = re.compile(separator)
    fine_split = re.split(sep_pat, text)
    separator_len = len_func(separator)

    if len(fine_split) <= 1:
        warnings.warn(f'No splits detected. Perhaps a problem with separator? ({repr(separator)})?')

    at_least_one_chunk = False

    # Combine the small pieces into medium size chunks
    # chunks will accumulate processed text chunks as we go along
    # curr_chunk will be a list of subchunks comprising the main, current chunk
    # back_overlap will be appended, once ready, to the end of the previous chunk (if any)
    # fwd_overlap will be prepended, once ready, to the start of the next chunk
    prev_chunk = ''
    curr_chunk, curr_chunk_len = [], 0
    back_overlap, back_overlap_len = None, 0  # None signals not yet gathering
    fwd_overlap, fwd_overlap_len = None, 0

    for s in fine_split:
        if not s: continue  # noqa E701
        split_len = len_func(s) + separator_len
        # Check for full back_overlap (if relevant, i.e. back_overlap isn't None)
        if back_overlap is not None and (back_overlap_len + split_len > chunk_overlap):  # noqa: F821
            prev_chunk.extend(back_overlap)
            back_overlap, back_overlap_len = None, 0

        # Will adding this split take us into overlap room?
        if curr_chunk_len + split_len > (chunk_size - chunk_overlap):
            fwd_overlap, fwd_overlap_len = [], 0  # Start gathering

        # Will adding this split take us over chunk size?
        if curr_chunk_len + split_len > chunk_size:
            # If so, complete current chunk & start a new one

            # fwd_overlap should be non-None at this point, so check empty
            if not fwd_overlap and curr_chunk:
                # If empty, look back to make sure there is some overlap
                fwd_overlap.append(curr_chunk[-1])

            prev_chunk = separator.join(prev_chunk)
            if prev_chunk:
                at_least_one_chunk = True
                yield prev_chunk
            prev_chunk = curr_chunk
            # fwd_overlap intentionally not counted in running chunk length
            curr_chunk, curr_chunk_len = fwd_overlap, 0
            back_overlap, back_overlap_len = [], 0  # Start gathering
            fwd_overlap, fwd_overlap_len = None, 0  # Stop gathering

        if fwd_overlap is not None:
            fwd_overlap.append(s)
            fwd_overlap_len += split_len

        if back_overlap is not None:
            back_overlap.append(s)
            back_overlap_len += split_len

        curr_chunk.append(s)
        curr_chunk_len += split_len

    # Done with the splits; use the final back_overlap, if any
    if back_overlap:
        prev_chunk.extend(back_overlap)

    if at_least_one_chunk:
        yield separator.join(prev_chunk)
    else:
        # Degenerate case where no splits found & chunk size too large; just one big chunk
        yield text

