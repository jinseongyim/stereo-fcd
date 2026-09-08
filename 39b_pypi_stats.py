# -*- coding: utf-8 -*-
"""PyPI 최근 다운로드 재수집 — 39_fcd_usage_evidence.py 가 HTTP 429 로 실패해
   fcd_usage_evidence.json 에 stats_error 만 남았던 부분을 backoff 로 다시 채운다.
   결과를 pypi_downloads.json 에 기록하고, 기존 evidence JSON 에도 병합한다."""
import io, json, os, time, urllib.request, urllib.error, datetime, sys

PKGS = ['fcd', 'fcd-torch']
UA = {'User-Agent': 'stereo-fcd-usage-evidence/1.0 (+manuscript reproducibility check)'}
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(name, tries=8):
    url = 'https://pypistats.org/api/packages/%s/recent' % name
    delay = 5.0
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode()), None
        except urllib.error.HTTPError as e:
            last = 'HTTP %s' % e.code
            if e.code != 429:
                return None, last
        except Exception as e:
            last = str(e)
        print('   %s: %s -> %.0fs 대기 후 재시도 (%d/%d)' % (name, last, delay, i + 1, tries))
        sys.stdout.flush()
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return None, last


def main():
    stamp = datetime.date.today().isoformat()
    out = {'accessed': stamp, 'source': 'https://pypistats.org/api/packages/<pkg>/recent',
           'window': 'rolling 30 days (last_month)', 'packages': {}}
    for p in PKGS:
        print('fetching', p)
        d, err = fetch(p)
        if d:
            out['packages'][p] = d['data']
            print('   OK  last_day=%(last_day)d last_week=%(last_week)d last_month=%(last_month)d' % d['data'])
        else:
            out['packages'][p] = {'error': err}
            print('   FAIL', err)
        time.sleep(8)   # 패키지 간 간격
    json.dump(out, open(os.path.join(HERE, 'pypi_downloads.json'), 'w'), indent=1)
    print('\n-> pypi_downloads.json')

    ev = os.path.join(HERE, 'fcd_usage_evidence.json')
    if os.path.exists(ev):
        e = json.load(io.open(ev, encoding='utf-8'))
        for rec in e.get('packages', []):
            got = out['packages'].get(rec['package'])
            if got and 'error' not in got:
                rec['downloads'] = got
                rec['downloads_accessed'] = stamp
                rec.pop('stats_error', None)
        json.dump(e, io.open(ev,'w',encoding='utf-8'), indent=1, ensure_ascii=False)
        print('-> fcd_usage_evidence.json 병합 완료')

    print('\n=== 본문에 써야 할 값 ===')
    for p in PKGS:
        v = out['packages'][p]
        print('  %-10s %s' % (p, ('%(last_month)s (month to %(d)s)' % dict(v, d=stamp))
                              if 'error' not in v else 'FAILED: %s' % v['error']))


if __name__ == '__main__':
    main()
