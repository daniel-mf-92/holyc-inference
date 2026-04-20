#!/usr/bin/env python3
from __future__ import annotations
import random, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from test_tokenizer_bpe_encode_prompt_checked import I64_MAX,TOKENIZER_BPE_ERR_BAD_PARAM,TOKENIZER_BPE_ERR_NULL_PTR,TOKENIZER_BPE_ERR_OVERFLOW,TOKENIZER_BPE_OK
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only import tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes import tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes

def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes(data,byte_len,io_cursor,prompt_nbytes,rank_left_tokens,rank_right_tokens,rank_values,rank_merged_tokens,rank_table_count,rank_table_capacity,max_piece_len,out_token_ids,out_token_count,out_required_token_capacity,out_required_merge_workspace_bytes):
    if data is None or io_cursor is None or out_token_ids is None or out_token_count is None or out_required_token_capacity is None or out_required_merge_workspace_bytes is None: return TOKENIZER_BPE_ERR_NULL_PTR
    if byte_len>I64_MAX or prompt_nbytes>I64_MAX or rank_table_count>I64_MAX or rank_table_capacity>I64_MAX or max_piece_len>I64_MAX: return TOKENIZER_BPE_ERR_OVERFLOW
    snapshot_cursor=io_cursor[0]
    if snapshot_cursor>byte_len: return TOKENIZER_BPE_ERR_BAD_PARAM
    staged_required_token_capacity=[out_required_token_capacity[0]]
    staged_required_merge_workspace_bytes=[out_required_merge_workspace_bytes[0]]
    snapshot_cursor_for_preflight=[snapshot_cursor]
    err=tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes(data,byte_len,snapshot_cursor_for_preflight,prompt_nbytes,rank_left_tokens,rank_right_tokens,rank_values,rank_merged_tokens,rank_table_count,rank_table_capacity,max_piece_len,staged_required_token_capacity,staged_required_merge_workspace_bytes)
    if err!=TOKENIZER_BPE_OK: return err
    if snapshot_cursor_for_preflight[0]!=io_cursor[0]: return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_required_token_capacity[0]!=prompt_nbytes: return TOKENIZER_BPE_ERR_BAD_PARAM
    if prompt_nbytes and max_piece_len>I64_MAX//prompt_nbytes: return TOKENIZER_BPE_ERR_OVERFLOW
    if staged_required_merge_workspace_bytes[0]!=prompt_nbytes*max_piece_len: return TOKENIZER_BPE_ERR_BAD_PARAM
    staged_cursor=[io_cursor[0]]; staged_count=[out_token_count[0]]; staged_required_after_commit=[out_required_token_capacity[0]]
    staged_capacity=max(1,prompt_nbytes)
    if staged_capacity>I64_MAX//4: return TOKENIZER_BPE_ERR_OVERFLOW
    staged_tokens=[0]*staged_capacity
    err=tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only(data,byte_len,staged_cursor,prompt_nbytes,rank_left_tokens,rank_right_tokens,rank_values,rank_merged_tokens,rank_table_count,rank_table_capacity,max_piece_len,staged_tokens,staged_count,staged_required_after_commit)
    if err!=TOKENIZER_BPE_OK: return err
    if staged_required_after_commit[0]!=staged_required_token_capacity[0]: return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_count[0]>staged_required_token_capacity[0]: return TOKENIZER_BPE_ERR_BAD_PARAM
    for i in range(staged_count[0]): out_token_ids[i]=staged_tokens[i]
    out_token_count[0]=staged_count[0]; out_required_token_capacity[0]=staged_required_after_commit[0]; out_required_merge_workspace_bytes[0]=staged_required_merge_workspace_bytes[0]; io_cursor[0]=staged_cursor[0]
    return TOKENIZER_BPE_OK

def explicit_checked_composition(*args): return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes(*args)

def test_source_contains_commit_only_required_bytes_wrapper():
    source=Path('src/tokenizer/bpe.HC').read_text(encoding='utf-8')
    sig='I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyRequiredBytes('
    assert sig in source
    body=source.split(sig,1)[1].split('I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens(',1)[0]
    assert 'TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytes(' in body
    assert 'TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnly(' in body
    assert 'staged_tokens = MAlloc(staged_alloc_bytes);' in body
    assert '*out_required_merge_workspace_bytes = staged_required_merge_workspace_bytes;' in body

def _tables():
    entries=sorted([(108,108,1,300),(200,300,2,400),(104,101,3,200),(400,111,0,500)],key=lambda x:(x[0],x[1]))
    return [e[0] for e in entries],[e[1] for e in entries],[e[2] for e in entries],[e[3] for e in entries]

def test_basic_success_and_overflow_no_partial():
    left,right,ranks,merged=_tables(); payload=list(b'hello world')
    toks=[0x11]*64; cnt=[0x22]; req=[0x33]; reqb=[0x44]; cur=[1]
    err=tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes(payload,len(payload),cur,5,left,right,ranks,merged,len(ranks),len(ranks),8,toks,cnt,req,reqb)
    assert err==TOKENIZER_BPE_OK and req==[5] and reqb==[40] and cnt[0]<=req[0]
    toks2=[9]*8; cnt2=[7]; req2=[6]; reqb2=[5]; cur2=[2]
    err=tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes(payload,len(payload),cur2,3,left,right,ranks,merged,len(ranks),len(ranks),I64_MAX,toks2,cnt2,req2,reqb2)
    assert err==TOKENIZER_BPE_ERR_OVERFLOW and toks2==[9]*8 and cnt2==[7] and req2==[6] and reqb2==[5] and cur2==[2]

def test_fuzz_parity():
    random.seed(20260421_824)
    for _ in range(1200):
        n=random.randint(0,80); payload=[random.randint(0,127) for _ in range(n)]; cur0=random.randint(0,n); pn=random.randint(0,n-cur0)
        rc=random.randint(0,16); rcap=rc+random.randint(0,2)
        rl=[random.randint(0,255) for _ in range(rc)]; rr=[random.randint(0,255) for _ in range(rc)]; rv=[random.randint(0,64) for _ in range(rc)]; rm=[random.randint(256,8192) for _ in range(rc)]
        if rc and random.random()<0.08: rl=None
        if rc and random.random()<0.08: rr=None
        if rc and random.random()<0.08: rv=None
        if rc and random.random()<0.08: rm=None
        mpl=random.randint(0,128)
        ta=[1]*128; tb=[1]*128; ca=[2]; cb=[2]; ra=[3]; rb=[3]; ba=[4]; bb=[4]; cura=[cur0]; curb=[cur0]
        ea=tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes(payload,n,cura,pn,rl,rr,rv,rm,rc,rcap,mpl,ta,ca,ra,ba)
        eb=explicit_checked_composition(payload,n,curb,pn,rl,rr,rv,rm,rc,rcap,mpl,tb,cb,rb,bb)
        assert ea==eb and ta==tb and ca==cb and ra==rb and ba==bb and cura==curb

if __name__=='__main__':
    test_source_contains_commit_only_required_bytes_wrapper(); test_basic_success_and_overflow_no_partial(); test_fuzz_parity(); print('ok')
