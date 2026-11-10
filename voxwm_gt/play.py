"""Keyboard/mouse play through real engine actions; optional synchronized GT recording."""
from contextlib import nullcontext
import numpy as np
from .session import GTSession
from .recording import Recorder


def keyboard_action(pg,wrapper,events):
    keys=pg.key.get_pressed();action=np.zeros(4,dtype=np.int64)
    names=[]
    for key,name in ((pg.K_w,'forward'),(pg.K_s,'backward'),(pg.K_a,'left'),(pg.K_d,'right')):
        if keys[key]: names.append(name)
    if 'forward' in names and 'backward' in names: names=[x for x in names if x not in ('forward','backward')]
    if 'left' in names and 'right' in names: names=[x for x in names if x not in ('left','right')]
    for key,name in ((pg.K_SPACE,'jump'),(pg.K_LSHIFT,'sneak'),(pg.K_LCTRL,'aux1')):
        if keys[key]: names.append(name)
    combo='+'.join(names)
    if combo in wrapper.actions[0]: action[0]=wrapper.actions[0].index(combo)+1
    # Arrow keys also work without mouse grab; click for dig/place.
    dx=dy=0
    for event in events:
        if event.type==pg.MOUSEMOTION and pg.event.get_grab(): dx+=event.rel[0];dy-=event.rel[1]
    dx+=int(keys[pg.K_RIGHT])-int(keys[pg.K_LEFT])
    dy+=int(keys[pg.K_UP])-int(keys[pg.K_DOWN])
    if dx: action[2]=wrapper.actions[2].index('mouse x+0.1' if dx>0 else 'mouse x-0.1')+1
    if dy: action[3]=wrapper.actions[3].index('mouse y+0.1' if dy>0 else 'mouse y-0.1')+1
    buttons=pg.mouse.get_pressed()
    if buttons[0]: action[1]=wrapper.actions[1].index('dig')+1
    elif buttons[2]:
        slot=next((i for i in range(1,10) if keys[getattr(pg,'K_'+str(i))]),1)
        action[1]=wrapper.actions[1].index('place+slot_'+str(slot))+1
    return action


def play(a):
    import pygame as pg
    if a.output and a.output.exists(): raise FileExistsError(a.output)
    pg.init()
    try:
        with GTSession(a.scene,a.seed,a.renderer,a.craftium_dir,a.world,a.mapgen) as s:
            screen=pg.display.set_mode((960,540));pg.display.set_caption('VoxWM GT: WASD, arrows/mouse, Space, Shift; Tab mouse grab; Esc exit')
            clock=pg.time.Clock()
            with Recorder(a.output,s) if a.output else nullcontext() as r:
                if r: r.append(np.zeros(4,dtype=np.int64))
                while True:
                    screen.blit(pg.transform.scale(pg.surfarray.make_surface(s.rgb.transpose(1,0,2)),screen.get_size()),(0,0));pg.display.flip()
                    events=pg.event.get()
                    if any(e.type==pg.QUIT or (e.type==pg.KEYDOWN and e.key==pg.K_ESCAPE) for e in events):
                        if r: r.record['stop_reason']='user_exit'
                        break
                    for e in events:
                        if e.type==pg.KEYDOWN and e.key==pg.K_TAB:
                            pg.event.set_grab(not pg.event.get_grab());pg.mouse.set_visible(not pg.event.get_grab());pg.mouse.get_rel()
                    action=keyboard_action(pg,s.env,events)
                    s.step(action)
                    if r: r.append(action)
                    if s.terminated or s.truncated:
                        if r: r.record['stop_reason']='native_engine_terminated'
                        break
                    clock.tick(24)
    finally: pg.quit()
