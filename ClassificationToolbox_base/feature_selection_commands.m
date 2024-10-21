function feature_selection_commands(command)

%This function deals with commands generated by the feature selection module

persistent methods;
if isempty(methods)
   methods = read_algorithms('Feature_selection.txt');
end

switch command
case 'OK'
   %OK Pressed
   h			= findobj('Tag', 'txtHiddenMethod');
   h1			= findobj('Tag', 'popMethod');
   chosen	= methods(get(h1, 'Value')).Name;         
   set(h, 'String', chosen)
   
   h			= findobj('Tag', 'txtHiddenParams');
   h1			= findobj('Tag', 'txtParameters');
   params 	= get(h1, 'String');
   set(h, 'String', params)

   set(gcf,'UserData',1)
		   
case 'Init'
    %Init of the classifier GUI
    h				= findobj('Tag', 'popMethod');
    set(h,'String',strvcat(methods(:).Name));
    chosen = strmatch('PCA',char(methods(:).Name));
    set(h,'Value',chosen);
    hLabel			= findobj('Tag', 'lblParameters');
    hBox				= findobj('Tag', 'txtParameters');
    set(hBox,'String',methods(chosen).Default);
    set(hBox,'Visible','on');
	 set(hLabel,'String',methods(chosen).Caption);
 	 set(hLabel,'Visible','on');
    
case 'Changed_method'
   h					= findobj(gcbf, 'Tag', 'popMethod');
   chosen        	= get(h, 'Value');
   
   hLabel			= findobj('Tag', 'lblParameters');
   hBox				= findobj('Tag', 'txtParameters');

   if ~isempty(chosen),
       switch methods(chosen).Field
       case 'S'
           set(hBox,'String',methods(chosen).Default);
           set(hBox,'Visible','on');
		     set(hLabel,'String',methods(chosen).Caption);
       	  set(hLabel,'Visible','on');
       case 'N'
          set(hLabel,'String','');
          set(hBox,'Visible','off');    
       end
   else
      set(hLabel,'String','');
      set(hBox,'Visible','off');    
   end
   
otherwise
   error('Unknown commands')
end
