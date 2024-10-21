function umtoggle(g)
if strcmp(get(g, 'Checked'),'on')
    set(g,'Checked', 'off');
else
    set(g,'Checked', 'on');
end
end